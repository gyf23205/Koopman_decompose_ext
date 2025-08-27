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


def koopman_loss(x, y, x_hat, latent_x, latent_y, p, model): # compute loss functions
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_hat, x) # Reconstruction loss (between x and x_hat)
    state_pred_loss = 0.0 # Prediction loss (up to p time steps)
    latent_pred_loss = 0.0 # Prediction loss in lifted space (up to p time steps)
    
    # time_steps, _ = x.size()
    # true_steps = 0

    # # pre compute K power
    # Ks = [torch.linalg.matrix_power(model.K.T, step - 1) for step in range(1, p + 1)]

    # for step in range(1, p + 1):
    #     if step >= time_steps: # if it reaches the data limit
    #         break
    #     true_steps += 1

    #     # True & lifted future state 
    #     true_future_state = x[step:, :]
    #     true_future_latent = z[step:, :] ######################################################

    #     # Predict future latent states using Koopman operator
    #     predicted_latent = z_pred[:-step, :]
    #     predicted_latent = torch.matmul(predicted_latent, Ks[step - 1])
        
    #     # Decoded predicted future states 
    #     predicted_state = model.decoder(predicted_latent)

    #     # State Prediction Loss
    #     state_pred_loss = state_pred_loss + mse_loss(predicted_state, true_future_state[:predicted_state.size(0), :])

    #     # Latent Prediction Loss
    #     latent_pred_loss = latent_pred_loss + mse_loss(predicted_latent, true_future_latent[:predicted_latent.size(0), :])
    
    predicted_latent = torch.matmul(latent_x, model.K.T)
    predicted_state = model.decoder(predicted_latent)

    # State Prediction Loss
    state_pred_loss = state_pred_loss + mse_loss(predicted_state, y)

    # Latent Prediction Loss
    latent_pred_loss = latent_pred_loss + mse_loss(predicted_latent, latent_y)

    # # Average prediction losses over p time steps
    # state_pred_loss /= true_steps
    # latent_pred_loss /= true_steps

    return recon_loss, state_pred_loss, latent_pred_loss

def convert_numpy_shape(input_data, return_tensor=True): # just to convert data shape
    reshaped_data = np.transpose(input_data, (2, 1, 0))  # (num_samples, time_steps, state_dim)

    if return_tensor:
        return torch.tensor(reshaped_data, dtype=torch.float32)
    else:
        return reshaped_data 

def compute_l_kae(kae, aug_input, aug_output, c1, c2, c3, p, device):
    # Concatenate input and output into a single augmented vector
    x = aug_input.to(device)  
    x = x.squeeze(1)   
    y = aug_output.to(device)  
    y = y.squeeze(1)   

    # print(x.device)
    # for name, param in kae.named_parameters():
    #     print(name, param.device)

    x_hat, latent_x, _ = kae(x)
    y_hat, latent_y, _ = kae(y)
    kae.compute_koopman_operator(latent_x, latent_y, device)

    # print(x.shape, x_hat.shape, latent_x.shape, latent_y.shape)

    # Compute losses
    recon_loss, state_pred_loss, koopman_pred_loss = koopman_loss(x, y, x_hat, latent_x, latent_y, p, kae)
    loss_kae = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss

    return loss_kae, latent_x

def compute_l_task(model, inputs, true_output, criterion):
    x = inputs.squeeze(1)
    y = true_output.squeeze(1)
    _,_, outputs = model(x)

    loss = criterion(outputs, y)

    return loss


def compute_theta_sub_all(kae, z, ko, n = 1):
    ko = torch.linalg.matrix_power(ko,n)
    eigvals, eigvec_left = torch.linalg.eig(ko)
    eigvec_left = eigvec_left.real.detach()
    eigvec_left_inv = torch.linalg.pinv(eigvec_left)
    v = (kae.decoder(eigvec_left_inv)).T
    phi = eigvec_left @ z[-1, :]
    param_sub_all = v @ torch.diag(phi)
    return param_sub_all, eigvals

def stt_decompose_reconstruction(kae, z, z_next, observable_dim, propagation = True):
    ko = kae.K
    if propagation:
        eigvals, eigvec_left = torch.linalg.eig(ko.T)
        eigvals = eigvals.conj().T
        eigvec_left = eigvec_left.conj().T
        eigvec_left_inv = torch.linalg.inv(eigvec_left)
        B = kae.decoder.linear.weight.detach().clone()
        B = B.to(torch.complex64)
        v = (B @ eigvec_left_inv) # kae dim x encoder dim

        phi = eigvec_left @ z.to(torch.complex64)
        # print(eigvals.shape, phi.shape, v.shape)
        # mode_output = v@phi@eigvals
        for i in range(0,observable_dim):
            if i == 0:
                temp = eigvals[0]*phi[0]*v[:,0]
            else:
                temp = temp + eigvals[i]*phi[i]*v[:,i]
        # mode_output = v*(eigvals*phi)
        mode_output = temp
    else:
        _, eigvec_left = torch.linalg.eig(ko.T)
        eigvec_left = eigvec_left.conj().T
        eigvec_left_inv = torch.linalg.inv(eigvec_left)
        B = kae.decoder.linear.weight.detach().clone()
        B = B.to(torch.complex64)
        v = (B @ eigvec_left_inv) # kae dim x encoder dim

        phi = eigvec_left @ z_next.to(torch.complex64)
        for i in range(0,observable_dim):
            if i == 0:
                temp = phi[0]*v[:,0]
            else:
                temp = temp + phi[i]*v[:,i]
        mode_output = temp
    return mode_output

def stt_decompose_mode(kae, z, z_next, observable_dim, mode_number, propagation = True):
    ko = kae.K
    if propagation:
        eigvals, eigvec_left = torch.linalg.eig(ko.T)
        eigvals = eigvals.conj().T
        eigvec_left = eigvec_left.conj().T
        eigvec_left_inv = torch.linalg.inv(eigvec_left)
        B = kae.decoder.linear.weight.detach().clone()
        B = B.to(torch.complex64)
        v = (B @ eigvec_left_inv) # kae dim x encoder dim

        phi = eigvec_left @ z.to(torch.complex64)
        temp = eigvals[mode_number]*phi[mode_number]*v[:,mode_number]
        mode_output = temp
    else:
        _, eigvec_left = torch.linalg.eig(ko.T)
        eigvec_left = eigvec_left.conj().T
        eigvec_left_inv = torch.linalg.inv(eigvec_left)
        B = kae.decoder.linear.weight.detach().clone()
        B = B.to(torch.complex64)
        v = (B @ eigvec_left_inv) # kae dim x encoder dim

        phi = eigvec_left @ z_next.to(torch.complex64)
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
