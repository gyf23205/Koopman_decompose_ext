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

from dataset.MNIST import MNIST
from dataset.FMNIST import FMNIST
from dataset.CIFAR10 import CIFAR10
from dataset.MNISTPerClass import MNISTPerClass
from dataset.FMNISTPerClass import FMNISTPerClass
from dataset.CIFAR10PerClass import CIFARPerClass

# import dill

# def collect_latent_states(model, x): # collect lifted (=latent) states 
#     x_hat, z = (x)
#     # print(z.shape)
#     return z

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

    x_hat, latent_x, _ = kae(x)
    y_hat, latent_y, _ = kae(y)
    kae.compute_koopman_operator(latent_x, latent_y)

    # print(x.shape, x_hat.shape, latent_x.shape, latent_y.shape)

    # Compute losses
    recon_loss, state_pred_loss, koopman_pred_loss = koopman_loss(x, y, x_hat, latent_x, latent_y, p, kae)
    loss_kae = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss

    return loss_kae, latent_x

def compute_l_classifier(model, images, labels, criterion_classifier, device):
    # Move tensors to device
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion_classifier(outputs, labels)
    return loss

def compute_theta_sub_all(kae, z, ko, n = 1):
    ko = torch.linalg.matrix_power(ko,n)
    eigvals, eigvec_left = torch.linalg.eig(ko)
    eigvec_left = eigvec_left.real.detach()
    eigvec_left_inv = torch.linalg.pinv(eigvec_left)
    # ##### REPLACE PINV
    # def fast_truncated_pinv(X, eps=1e-6):
    #     U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    #     S_inv = torch.where(S > eps, 1.0 / S, torch.zeros_like(S))
    #     return Vh.T @ torch.diag(S_inv) @ U.T
    # eigvec_left_inv = fast_truncated_pinv(eigvec_left)
    ###################################
    v = (kae.decoder(eigvec_left_inv)).T
    phi = eigvec_left @ z[-1, :]
    param_sub_all = v @ torch.diag(phi)
    return param_sub_all, eigvals

def stt_decompose_each(kae, z, mode_number):
    ko = kae.K
    eigvals, eigvec_left = torch.linalg.eig(ko)
    eigvec_left_inv = torch.linalg.pinv(eigvec_left)
    B = kae.decoder.linear.weight.detach().clone()
    B = B.to(torch.complex64)
    v = (B@eigvec_left_inv) # kae dim x encoder dim
    v_use = v[:,mode_number]

    phi = eigvec_left @ z.to(torch.complex64)
    mode_output = v_use * phi[mode_number]
    mode_output = mode_output*eigvals[mode_number]
    # print(np.shape(B))
    # print(np.shape(eigvec_left))
    # print(np.shape(z))
    # print(np.shape(phi))
    # print(np.shape(v))
    # print(np.shape(v_use))
    # print(np.shape(eigvals))
    # print(np.shape(mode_output))
    # print(np.shape(stt_decomp*eigvals.T))
    return mode_output

def stt_decompose_reconstruction(kae, z, z_next, kae_size, propagation = True):
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
        for i in range(0,kae_size):
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
        for i in range(0,kae_size):
            if i == 0:
                temp = phi[0]*v[:,0]
            else:
                temp = temp + phi[i]*v[:,i]
        mode_output = temp
    return mode_output

def stt_decompose_mode(kae, z, z_next, kae_size, mode_number, propagation = True):
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


# def compute_theta_sub_all_steps(kae, z, ko, n):
#     ko = torch.linalg.matrix_power(ko,n)
#     eigvals, eigvec_left = torch.linalg.eig(ko)
#     eigvec_left = eigvec_left.real.detach()
#     eigvec_left_inv = torch.linalg.pinv(eigvec_left)
#     v = (kae.decoder(eigvec_left_inv)).T
#     phi = eigvec_left @ z[-1, :]
#     param_sub_all = v @ torch.diag(phi)
#     return param_sub_all, eigvals

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
        # # images = images.to(device)
        # # labels = labels.to(device)
        # _,_,outputs = model(images)
        # outputs = outputs[:, :num_classes]
        # predicted = torch.argmax(outputs, dim=1)
        # correct = (predicted == labels).sum().item()
        # total = labels.size(0)
        # accuracy = 100.0 * correct / total
        _,_,outputs = model(images)
        outputs = outputs[:, :num_classes]
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100.0 * correct / total
    return accuracy

def load_dataset(target_dataset, batch_size):
    if target_dataset == 'MNIST':
        dataset_in_use_per_class = MNISTPerClass(batch_size=batch_size)
        dataset_in_use = MNIST(batch_size=batch_size)
        # max_param_stack = 2 ** 9
        # hidden_c = 16

    elif target_dataset == 'FMNIST':
        dataset_in_use_per_class = FMNISTPerClass(batch_size=batch_size)
        dataset_in_use = FMNIST(batch_size=batch_size)
        # max_param_stack = 2 ** 9
        # hidden_c = 16

    elif target_dataset == 'CIFAR10':
        dataset_in_use_per_class = CIFARPerClass(batch_size=batch_size)
        dataset_in_use = CIFAR10(batch_size=batch_size)
        # max_param_stack = 2 ** 9
        # hidden_c = 16

    elif target_dataset == 'humanoid':
        print('NO KAE_CLASSIFIER NOW')
        dataset_in_use_per_class, dataset_in_use = None, None

    return dataset_in_use_per_class, dataset_in_use

# # Generalized version
# def compute_l_classifier_within(
#     param_sub,
#     images,
#     labels,
#     criterion_classifier,
#     classifier_shapes,
#     param_shapes,
#     device
# ):
#     """
#     Args:
#         param_sub (torch.Tensor): Flattened model parameters
#         images (torch.Tensor): Input images or observations
#         labels (torch.Tensor): Target labels
#         criterion_classifier (nn.Module): Loss function (e.g., nn.CrossEntropyLoss)
#         classifier_shapes (List[Dict]): Model structure
#         param_shapes (List[torch.Size]): Parameter shapes
#         device (torch.device): CUDA or CPU

#     Returns:
#         torch.Tensor: Classification loss
#     """
#     images = images.to(device)
#     labels = labels.to(device)
    
#     if images.dim() == 2 and images.size(1) == 28 * 28:
#         images = images.view(-1, 28 * 28)  # flatten for MLP
#     elif images.dim() == 4:
#         pass  # leave CNN input as is
#     else:
#         raise ValueError(f"Unexpected input shape: {images.shape}")
    
#     outputs = classifier_sub(images, param_sub, classifier_shapes, param_shapes)
#     loss = criterion_classifier(outputs, labels)
#     return loss

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

def classifier_sub(x, p_vec, classifier_shapes, param_shapes):
    """
    Args:
        x (torch.Tensor): Input tensor
        p_vec (torch.Tensor): Flattened parameter vector
        classifier_shapes (List[Dict]): Layer structure from extract_model_structure_and_shapes()
        param_shapes (List[torch.Size]): Flattened parameter shapes (ordered)

    Returns:
        torch.Tensor: Output after applying the model
    """
    return functional_forward(x, p_vec, classifier_shapes, param_shapes)


def extract_model_structure_and_shapes(model: nn.Module):
    traced = symbolic_trace(model)
    modules = dict(model.named_modules())
    model_structure = []
    param_shapes = []

    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        layer = modules[node.target]
        layer_type = type(layer)

        if isinstance(layer, nn.Conv2d):
            model_structure.append({
                "type": "conv2d",
                "params": [layer.weight.shape, layer.bias.shape],
                "stride": layer.stride,
                "padding": layer.padding,
            })
            param_shapes.extend([layer.weight.shape, layer.bias.shape])
        elif isinstance(layer, nn.Linear):
            model_structure.append({
                "type": "linear",
                "params": [layer.weight.shape, layer.bias.shape],
            })
            param_shapes.extend([layer.weight.shape, layer.bias.shape])
        elif isinstance(layer, nn.ReLU):
            model_structure.append({"type": "relu"})
        elif isinstance(layer, nn.Tanh):
            model_structure.append({"type": "tanh"})
        elif isinstance(layer, nn.MaxPool2d):
            model_structure.append({
                "type": "maxpool2d",
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
            })
        elif isinstance(layer, nn.Flatten):
            model_structure.append({"type": "flatten"})
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    return model_structure, param_shapes

def flatten_model_params(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters())

def functional_forward(x: torch.Tensor, p_vec: torch.Tensor, model_structure, param_shapes):
    device = x.device
    idx = 0
    for layer in model_structure:
        layer_type = layer["type"]

        if "params" in layer:
            layer_params = []
            for shape in layer["params"]:
                numel = torch.prod(torch.tensor(shape, device=device)).item()
                param = p_vec[idx:idx + numel].view(shape).to(device)
                layer_params.append(param)
                idx += numel
        else:
            layer_params = []

        if layer_type == "conv2d":
            weight, bias = layer_params
            x = F.conv2d(x, weight, bias, stride=layer["stride"], padding=layer["padding"])
        elif layer_type == "linear":
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            weight, bias = layer_params
            x = F.linear(x, weight, bias)
        elif layer_type == "relu":
            x = F.relu(x)
        elif layer_type == "tanh":
            x = torch.tanh(x)
        elif layer_type == "maxpool2d":
            x = F.max_pool2d(x, kernel_size=layer["kernel_size"], stride=layer["stride"])
        elif layer_type == "flatten":
            x = x.view(x.size(0), -1)
        else:
            raise NotImplementedError(f"Unsupported layer: {layer_type}")

    return x