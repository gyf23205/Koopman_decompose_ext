from tqdm import tqdm
import math
import pickle
import scipy
from torch.utils.data import DataLoader, TensorDataset
import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.fx import symbolic_trace
import matplotlib.pyplot as plt

def relative_mse_loss(x_hat, x, eps=1e-8):
    """
    Scale-free MSE that propagates gradients properly.
    Computes ((x_hat - x) / (|x| + eps))^2 averaged over all elements.
    """
    rel_err = (x_hat - x) / (torch.abs(x) + eps)
    return torch.mean(rel_err ** 2)

def cosine_mse_loss(x_hat, x):
    cos_sim = F.cosine_similarity(x_hat, x, dim=-1)
    return torch.mean(1 - cos_sim)  # equivalent to angular error

def vector_norm_mse_loss(x_hat, x, eps=1e-8, scale=1.0):
    x_hat, x = map(torch.as_tensor, (x_hat, x))
    x_hat, x = x_hat / (x_hat.norm(dim=-1, keepdim=True) + eps), x / (x.norm(dim=-1, keepdim=True) + eps)
    return ((x_hat - x) ** 2).mean() * scale

def log_safe(x, eps=1e-8):
    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)

def log_mse_loss(y_hat, y):
    return torch.mean((log_safe(y_hat) - log_safe(y))**2)

def barron_loss(y_hat, y, alpha=0.5, c=1.0, eps=1e-8):
    x2 = ((y_hat - y) / c) ** 2
    alpha_t = torch.as_tensor(alpha, dtype=y_hat.dtype, device=y_hat.device)
    two = torch.as_tensor(2.0, dtype=y_hat.dtype, device=y_hat.device)
    abs_term = torch.abs(alpha_t - two) + eps
    if alpha == 2:
        loss = 0.5 * x2
    else:
        loss = (abs_term / alpha_t) * (((x2 / abs_term) + 1.0).pow(alpha_t / 2.0) - 1.0)
    return loss.mean()

def koopman_loss(x, x_hat, latent_x, y_seq_states, y_seq_latents, p, action_dim, model):
    """
    Koopman losses with multi-step supervision.
    
    Args:
        x             : [B, pad_dim]  current state
        x_hat         : [B, pad_dim]  reconstruction of x
        latent_x      : [B, obv_dim]
        y_seq_states  : [B, p, pad_dim] true states for steps 1..m
        y_seq_latents : [B, p, obv_dim] true latents for steps 1..m
        p             : rollout horizon (p=1 = one step)
        model         : Koopman AE (must have model.K and model.decoder)

    Returns:
        recon_loss, state_pred_loss, latent_pred_loss
    """

    # Reconstruction loss (x vs x_hat)
    mse_loss = nn.MSELoss()
    # recon_loss = barron_loss(x_hat, x)
    recon_loss = mse_loss(x_hat, x)
    state_pred_loss = 0.0
    latent_pred_loss = 0.0
    action_loss = 0.0
    B,_, _ = y_seq_states.shape

    # Precompute K powers
    Ks = [torch.linalg.matrix_power(model.K, step) for step in range(1, p + 1)]

    # Roll forward in latent space
    for k in range(p):
        pred_lat_k = latent_x @ (Ks[k].T)              # [B, z]
        # pred_lat_k = (Ks[k])@latent_x.T   
        pred_y_k   = model.decoder(pred_lat_k)     # [B, l]

        # state_pred_loss  = state_pred_loss + barron_loss(pred_y_k,   y_seq_states[:, k, :])
        state_pred_loss  = state_pred_loss + mse_loss(pred_y_k,   y_seq_states[:, k, :])
        # latent_pred_loss = latent_pred_loss + barron_loss(pred_lat_k, y_seq_latents[:, k, :])
        latent_pred_loss = latent_pred_loss + mse_loss(pred_lat_k, y_seq_latents[:, k, :])
        action_loss = action_loss + mse_loss(pred_y_k[:,:action_dim], y_seq_states[:, k, :action_dim])


    state_pred_loss  /= p
    latent_pred_loss /= p
    action_loss /= p
    

    return recon_loss, state_pred_loss, latent_pred_loss, action_loss

def convert_numpy_shape(input_data, return_tensor=True): # just to convert data shape
    reshaped_data = np.transpose(input_data, (2, 1, 0))  # (num_samples, time_steps, state_dim)

    if return_tensor:
        return torch.tensor(reshaped_data, dtype=torch.float32)
    else:
        return reshaped_data 

def compute_l_kae(kae, aug_input, aug_output, c1, c2, c3, p, 
            device, aug_input_all, aug_output_all, inner, batch_size, action_dim):
    """
    Multi-step Koopman AE loss.
    - p = function-wise rollout horizon (p=1 = one-step)
    - aug_input_all / aug_output_all: full tensors [N,1,l] on CPU, ordered (shuffle=False)
    - inner: batch index from enumerate(train_loader)
    """
    B = aug_input.size(0)
    start = inner * batch_size
    end   = start + B
    # print(aug_input_all.size())
    # print(aug_output_all.size())

    # Current input batch
    x = aug_input.to(device).squeeze(1)        # [B, l]
    x_hat, latent_x, _ = kae(x)

    # Collect up to p true future states from aug_output_all
    y_seq_states = []
    y_seq_latents = []
    for k in range(p):
        if end + k > len(aug_output_all):   # don’t run past dataset
            break
        yk = aug_output_all[start+k:end+k].to(device).squeeze(1)  # [B, l]
        _, latent_yk, _ = kae(yk)
        y_seq_states.append(yk)
        y_seq_latents.append(latent_yk)
    
    # y_seq_states: [1, batch_size, pad_dim]
    # y_seq_stay_seq_latentstes: [1, batch_size, obv_dim]

    if not y_seq_states:
        raise RuntimeError("No future states available for multi-step supervision.")
    
    if len(y_seq_states) < p:
        last_y  = y_seq_states[-1]
        last_ly = y_seq_latents[-1]
        for _ in range(p - len(y_seq_states)):
            y_seq_states.append(last_y)
            y_seq_latents.append(last_ly)

    y_seq_states  = torch.stack(y_seq_states,  dim=1)  # [B, p, pad_dim]
    y_seq_latents = torch.stack(y_seq_latents, dim=1)  # [B, p, obv_siz]

    # print(y_seq_latents.size())

    # Update Koopman operator from first step
    # _, latent_x_all, _ = kae(aug_input_all)
    # _, latent_y_all, _ = kae(aug_output_all)
    # kae.compute_koopman_operator(latent_x_all, latent_y_all, device)

    # Compute losses
    recon_loss, state_pred_loss, koopman_pred_loss, loss_action = koopman_loss(
        x, x_hat, latent_x, y_seq_states, y_seq_latents, p, action_dim, kae
    )

    loss_kae = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss    
    return loss_kae, loss_action

# def compute_l_dist(observable_dim, current_eig, loss_dist_mc_sample_num, obs_dim, state_bound_lo, state_bound_hi, 
#                    padded_dimension, p, model, device):
    
#     kae = model
#     loss_dist = 0
#     tol = 1e-12

#     def is_real(z, tol=1e-12):
#         return abs(z.imag) <= tol

#     rep_idxs = []
#     for i, lam in enumerate(current_eig):
#         if lam.imag > tol:
#             rep_idxs.append(i)
#         elif is_real(lam, tol):
#             rep_idxs.append(i)

#     print("loop start")

#     for a, i in enumerate(rep_idxs):
#         for j in rep_idxs[a+1:]:
#             loss_dist_temp = torch.zeros(1).to(device)
#             loss_dist_temp_1 = torch.zeros(1).to(device)
#             loss_dist_temp_2 = torch.zeros(1).to(device)
#             # print(loss_dist_temp, loss_dist_temp_1, loss_dist_temp_2)

#             for k in range(0,loss_dist_mc_sample_num):
#                 random_input_loss_dist = torch.rand(1, obs_dim, device=device) * (state_bound_hi - state_bound_lo) + state_bound_lo
#                 pad_in_loss_dist = torch.ones(random_input_loss_dist.size(0), padded_dimension - obs_dim, device=device)
#                 aug_input_loss_dist = torch.cat([random_input_loss_dist, pad_in_loss_dist], dim=1)
#                 _,z_loss_dist,_ = kae(aug_input_loss_dist)

#                 loss_dist_temp_1 = stt_decompose_mode(kae, z_loss_dist.T,_, i, p, propagation = True)
#                 loss_dist_temp_2 = stt_decompose_mode(kae, z_loss_dist.T,_, j, p, propagation = True)
#                 loss_dist_temp = loss_dist_temp + loss_dist_temp_1*loss_dist_temp_2.conj()

#             loss_dist = loss_dist + (loss_dist_temp/loss_dist_mc_sample_num)

#     loss_dist = torch.abs(torch.sum(loss_dist).real)    

#     print("loop ends")

#     return loss_dist

def compute_l_dist(observable_dim, current_eig, loss_dist_mc_sample_num, obs_dim,
                   state_bound_lo, state_bound_hi, padded_dimension, p, model, device):

    kae = model
    tol = 1e-12

    # --- eigenvalue filtering ---
    rep_idxs = []
    for i, lam in enumerate(current_eig):
        if lam.imag > tol or abs(lam.imag) <= tol:
            rep_idxs.append(i)
    if len(rep_idxs) == 0:
        return torch.zeros(1, device=device, dtype=torch.float32)

    # --- preallocation ---
    mc = loss_dist_mc_sample_num
    rand = torch.empty(mc, obs_dim, device=device).uniform_()
    rand.mul_(state_bound_hi - state_bound_lo).add_(state_bound_lo)
    pad = torch.ones(mc, padded_dimension - obs_dim, device=device)
    aug_input = torch.cat((rand, pad), dim=1)

    # single forward call
    _, z_all, _ = kae(aug_input)
    z_all_T = z_all.T.contiguous()  # [latent_dim, mc]

    # preallocate for accumulation
    loss_dist = torch.zeros((), device=device, dtype=torch.complex64)
    zero_scalar = torch.zeros((), device=device, dtype=torch.complex64)

    # use local vars to reduce lookups
    sdm = stt_decompose_mode
    rep_len = len(rep_idxs)
    rng = range
    kaeref = kae

    # --- nested loops (same logic, faster execution) ---
    for a in rng(rep_len):
        i = rep_idxs[a]
        for j in rep_idxs[a + 1:]:
            acc = zero_scalar.clone()
            # inner loop optimized with tensor slicing, no realloc
            for k in rng(mc):
                z_loss_dist = z_all_T[:, k:k + 1]  # shape [latent_dim, 1]
                l1 = sdm(kaeref, z_loss_dist, None, i, p, propagation=True)
                l2 = sdm(kaeref, z_loss_dist, None, j, p, propagation=True)
                acc += (l1 * l2.conj()).sum()
            loss_dist += acc / mc

    # --- final reduction to real scalar ---
    loss_dist = torch.abs(loss_dist).real.to(torch.float32)
    return loss_dist


def compute_l_task(model, inputs, true_output, criterion, max_reward, device):
    x = inputs.squeeze(1).to(device)
    y = true_output.squeeze(1).to(device)
    _,_, outputs = model(x)

    loss = criterion(outputs, y)

    # reward = test_cartpole_kae_function(model, hidden_k, padded_dimension, p, device, mode_number = -1, num_episodes = num_episodes, save_imgs = True)
    # loss = criterion(reward, max_reward)

    return loss

def compute_theta_sub_all(kae, z, ko, n = 1):
    ko = torch.linalg.matrix_power(ko,n)
    eigvals, eigvec_left = torch.linalg.eig(ko.T)
    eigvec_left = eigvec_left.real.detach()
    eigvec_left_inv = torch.linalg.pinv(eigvec_left)
    v = (kae.decoder(eigvec_left_inv)).T
    phi = eigvec_left @ z[-1, :]
    param_sub_all = v @ torch.diag(phi)
    return param_sub_all, eigvals

# def stt_decompose_reconstruction(kae, z, z_next, observable_dim, p, propagation = True):
#     ko = kae.K
#     if propagation:
#         # eigvals, eigvec_left = torch.linalg.eig(ko.T)
#         eigvals, eigvec_left = torch.linalg.eig(ko.T)
#         eigvals = eigvals.conj().T
#         eigvec_left = eigvec_left.conj().T
#         eigvec_left_inv = torch.linalg.inv(eigvec_left)
#         B = kae.decoder.linear.weight.detach().clone()
#         B = B.to(torch.complex64)
#         v = (B @ eigvec_left_inv) # kae dim x encoder dim

#         phi = eigvec_left @ z.to(torch.complex64)
#         # print(eigvals.shape, phi.shape, v.shape)
#         # mode_output = v@phi@eigvals
#         for i in range(0,observable_dim):
#             if i == 0:
#                 temp = (eigvals[0]**p)*phi[0]*v[:,0]
#             else:
#                 temp = temp + (eigvals[i]**p)*phi[i]*v[:,i]
#         # mode_output = v*(eigvals*phi)
#     else:
#         # _, eigvec_left = torch.linalg.eig(ko.T)
#         _, eigvec_left = torch.linalg.eig(ko.T)
#         eigvec_left = eigvec_left.conj().T
#         eigvec_left_inv = torch.linalg.inv(eigvec_left)
#         B = kae.decoder.linear.weight.detach().clone()
#         B = B.to(torch.complex64)
#         v = (B @ eigvec_left_inv) # kae dim x encoder dim

#         phi = eigvec_left @ z_next.to(torch.complex64)
#         for i in range(0,observable_dim):
#             if i == 0:
#                 temp = phi[0]*v[:,0]
#             else:
#                 temp = temp + phi[i]*v[:,i]
#     mode_output = temp
#     return mode_output

def stt_decompose_reconstruction_isaac(kae, z, z_next, observable_dim, p, act_dim, propagation = True):
    """
    Batched Koopman reconstruction with modal summation.
      z, z_next : [B, observable_dim]
      kae.K     : [observable_dim, observable_dim]
      kae.decoder.linear.weight : [D, observable_dim]
    Returns mode_output : [B, D]
    """

    device = z.device

    # eigendecomposition of Kᵀ → left eigenvectors of K are rows of L
    # eigvals, eigvec_left = torch.linalg.eig(kae.K.T.to(torch.complex64))
    _, eigvec_left = torch.linalg.eig(kae.K.T.to(torch.complex64))

    # G.T A = LeftEig G.T
    # _, G = torch.linalg.eig(A.T)
    # G.'T' = [l1.T, l2.T, ...], each l_i = row vec

    eigvals, _ = torch.linalg.eig(kae.K.to(torch.complex64))
    # eigvals = eigvals.conj().T                     # column eigenvalues of K
    eigvec_left = eigvec_left.T           # [observable_dim, observable_dim]
    eigvec_left_inv = torch.linalg.inv(eigvec_left)


    ##################################################################

    # decoder linear matrix
    B = kae.decoder.linear.weight.detach().clone().to(torch.complex64).to(device)  # [D, observable_dim]
    v = B @ eigvec_left_inv                    # [D, observable_dim]

    # eq (21) → [B, observable_dim]
    z = z.to(torch.complex64)
    # print('z')
    # print(z.size())
    # print('eigvec_left')
    # print(eigvec_left.size())
    psi = z @ eigvec_left.T
    psi = psi
    # print('psi')
    # print(psi.size())
    # psi = torch.einsum("ij,bj->bi", eigvec_left, z)

    # compute each term v[:, i] * (λ_i**p * φ_bi) and sum across i
    # eig_pow = eigvals[:observable_dim] ** (p if propagation else 1)
    # mode_output = torch.einsum("bi,di->bd", psi * eig_pow, v[:, :observable_dim])

    eig_pow = eigvals ** (p if propagation else 1)
    # print('eig_pow')
    # print(eig_pow.size())
    mode_output = torch.einsum("bi,di->bd", psi * eig_pow, v)
    # print('mode_ouput')
    # print(mode_output.size())
    return mode_output[:, :act_dim].real

# # def stt_decompose_mode(kae, z, z_next, mode_number, p, propagation = True, conjugate = False):
# #     ko = kae.K
# #     eigvals, eigvec_left = torch.linalg.eig(ko.T)
# #     eigvals = eigvals.conj().T
# #     eigvec_left = eigvec_left.conj().T
# #     eigvec_left_inv = torch.linalg.inv(eigvec_left)

# #     if propagation:
# #         B = kae.decoder.linear.weight.detach().clone().to(torch.complex64)
# #         v = (B @ eigvec_left_inv) # kae dim x encoder dim

# #         phi = eigvec_left @ z.to(torch.complex64)
# #         if conjugate:
# #             temp = ((eigvals[mode_number]**p)*phi[mode_number]*v[:,mode_number]).conj()
# #         else:
# #             temp = (eigvals[mode_number]**p)*phi[mode_number]*v[:,mode_number]
# #     else:
# #         B = kae.decoder.linear.weight.detach().clone().to(torch.complex64)
# #         v = (B @ eigvec_left_inv) # kae dim x encoder dim

# #         phi = eigvec_left @ z_next.to(torch.complex64)
# #         if conjugate:
# #             temp = (phi[mode_number]*v[:,mode_number]).conj()
# #         else:
# #             temp = phi[mode_number]*v[:,mode_number]
    
# #     mode_output = temp
# #     return mode_output

def stt_decompose_mode(kae, z, z_next, mode_number, p, propagation = True, conjugate = False):
    ko = kae.K
    # eigvals, eigvec_left = torch.linalg.eig(ko.T)
    eigvals, eigvec_left = torch.linalg.eig(ko.T)
    eigvals = eigvals.conj().T
    eigvec_left = eigvec_left.conj().T

    # use linear solve instead of explicit inverse
    if propagation:
        B = kae.decoder.linear.weight.to(torch.complex64)
        # v = (B @ eigvec_left_inv)
        v = torch.linalg.solve(eigvec_left.T, B.T).T

        phi = eigvec_left @ z.to(torch.complex64)
        if conjugate:
            temp = ((eigvals[mode_number]**p)*phi[mode_number]*v[:,mode_number]).conj()
        else:
            temp = (eigvals[mode_number]**p)*phi[mode_number]*v[:,mode_number]
    else:
        B = kae.decoder.linear.weight.to(torch.complex64)
        v = torch.linalg.solve(eigvec_left.T, B.T).T

        phi = eigvec_left @ z_next.to(torch.complex64)
        if conjugate:
            temp = (phi[mode_number]*v[:,mode_number]).conj()
        else:
            temp = phi[mode_number]*v[:,mode_number]

    mode_output = temp
    return mode_output


def test_classifier(model, test_loader, device):
    model.eval()  # evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

def test_classifier_return(model, images, labels, device, num_classes=None):
    model.eval()
    with torch.no_grad():
        _,_,outputs = model(images)
        outputs = outputs[:, :num_classes]
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100.0 * correct / total
    return accuracy


def compute_l_classifier_within(
    images,
    labels,
    criterion_classifier,
    model,
    num_class,
    device
):
    images = images.to(device)
    labels = labels.to(device)
    
    if images.dim() == 2 and images.size(1) == 28 * 28:
        images = images.view(-1, 28 * 28)  # flatten for MLP
    elif images.dim() == 4:
        pass  # leave CNN input as is
    else:
        raise ValueError(f"Unexpected input shape: {images.shape}")
    
    _,_,outputs = model(images) 
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    if num_class is not None and outputs.size(1) > num_class:
        outputs = outputs[:, :num_class]

    loss = criterion_classifier(outputs, labels)
    return loss
