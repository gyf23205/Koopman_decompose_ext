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

# from dataset.MNIST import MNIST
# from dataset.FMNIST import FMNIST
# from dataset.CIFAR10 import CIFAR10
# from dataset.MNISTPerClass import MNISTPerClass
# from dataset.FMNISTPerClass import FMNISTPerClass
# from dataset.CIFAR10PerClass import CIFARPerClass

# import dill

def collect_latent_states(model, param_traj): # collect lifted (=latent) states 
    z = model.encoder(param_traj)
    latents = z[:-1, :]  # States at time t
    latents_next = z[1:, :]  # States at time t+1
    
    # latent_X = torch.cat(latent_X_list, dim=0)
    # latent_Y = torch.cat(latent_Y_list, dim=0)
    
    return latents, latents_next

def koopman_loss(x, x_hat, z, z_pred, p, model): # compute loss functions
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_hat, x) # Reconstruction loss (between x and x_hat)
    state_pred_loss = 0.0 # Prediction loss (up to p time steps)
    latent_pred_loss = 0.0 # Prediction loss in lifted space (up to p time steps)
    time_steps, _ = x.size()

    true_steps = 0

    # pre compute K power
    Ks = [torch.linalg.matrix_power(model.K.T, step - 1) for step in range(1, p + 1)]

    for step in range(1, p + 1):
        if step >= time_steps: # if it reaches the data limit
            break
        true_steps += 1

        # True & lifted future state 
        true_future_state = x[step:, :]
        true_future_latent = z[step:, :] ######################################################

        # Predict future latent states using Koopman operator
        predicted_latent = z_pred[:-step, :]
        predicted_latent = torch.matmul(predicted_latent, Ks[step - 1])
        
        # Decoded predicted future states 
        predicted_state = model.decoder(predicted_latent)

        # State Prediction Loss
        state_pred_loss = state_pred_loss + mse_loss(predicted_state, true_future_state[:predicted_state.size(0), :])

        # Latent Prediction Loss
        latent_pred_loss = latent_pred_loss + mse_loss(predicted_latent, true_future_latent[:predicted_latent.size(0), :])


    # Average prediction losses over p time steps
    state_pred_loss /= true_steps
    latent_pred_loss /= true_steps

    return recon_loss, state_pred_loss, latent_pred_loss

def convert_numpy_shape(input_data, return_tensor=True): # just to convert data shape
    reshaped_data = np.transpose(input_data, (2, 1, 0))  # (num_samples, time_steps, state_dim)

    if return_tensor:
        return torch.tensor(reshaped_data, dtype=torch.float32)
    else:
        return reshaped_data 

def compute_l_kae(kae, params_snapshots, c1, c2, c3, p, device):
    x = params_snapshots.to(device)
    latents, latents_next = collect_latent_states(kae, x)
    kae.compute_koopman_operator(latents, latents_next)
    x_hat, z, z_pred = kae(x)
    recon_loss, state_pred_loss, koopman_pred_loss = koopman_loss(x, x_hat, z, z_pred, p, kae)
    loss_kae = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss # + c4*k_norm_loss      
    return loss_kae, z

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

def test_classifier_return(model, test_loader, device):
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
    return accuracy

# Generalized version
def compute_l_classifier_within(
    param_sub,
    images,
    labels,
    criterion_classifier,
    classifier_shapes,
    param_shapes,
    device
):
    """
    Args:
        param_sub (torch.Tensor): Flattened model parameters
        images (torch.Tensor): Input images or observations
        labels (torch.Tensor): Target labels
        criterion_classifier (nn.Module): Loss function (e.g., nn.CrossEntropyLoss)
        classifier_shapes (List[Dict]): Model structure
        param_shapes (List[torch.Size]): Parameter shapes
        device (torch.device): CUDA or CPU

    Returns:
        torch.Tensor: Classification loss
    """
    images = images.to(device)
    labels = labels.to(device)
    
    if images.dim() == 2 and images.size(1) == 28 * 28:
        images = images.view(-1, 28 * 28)  # flatten for MLP
    elif images.dim() == 4:
        pass  # leave CNN input as is
    else:
        raise ValueError(f"Unexpected input shape: {images.shape}")
    
    outputs = classifier_sub(images, param_sub, classifier_shapes, param_shapes)
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