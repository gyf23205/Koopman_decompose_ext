import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.Autoencoder_functions import *
from models.Autoencoder import KoopmanAutoencoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import *
import wandb

if __name__ == "__main__":
    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently using... '+str(device))

    # Set training to be deterministic
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Hyperparameters
    num_epochs = 2000
    lr_kae = 1e-5
    kae_coef = 0.75
    c1, c2, c3 = 1, 1, 1 # KAE loss parameters
    batch_size = 128
    hidden_dim = 64
    p = 5

    # Wandb initialization
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    wandb.init(project="koopman_ext", name='cartpole_mlp', config={
        "num_epochs": num_epochs,
        "lr_kae": lr_kae,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "p": p
    })

    # Load the saved parameter trajectory
    param_traj = np.load("saved_trajectory/param_trajectory_cartpole_mlp.npy")
    state_dim = param_traj.shape[1]
    dataset_traj = DataLoader(TensorDataset(torch.tensor(param_traj, dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=False)

    # Build the Koopman Autoencoder and optimizer
    kae = KoopmanAutoencoder(state_dim, hidden_dim).to(device)
    kae.train()
    optimizer_kae = optim.Adam(kae.parameters(), lr=lr_kae)

    for epoch in (tq_var := tqdm(range(num_epochs), desc="Training")):
        running_loss = 0.0
        running_classifier_loss = 0.0

        # Each batch
        for inner, (params,) in enumerate(dataset_traj):  # Unpack the tuple correctly
            loss_kae_classifier = 0.0
            lose_kae = 0.0

            # loss_kae
            loss_kae, z = compute_l_kae(kae, params, c1, c2, c3, p, device)

            # Task specific loss
            # N_O = z.shape[-1]
            # param_sub_all, eigvals = compute_theta_sub_all(kae, z, kae.K)
            # param_sub_kae = torch.sum(param_sub_all[:, :hidden_dim], dim=1)  # Why the number of dominant modes is the same number of hidden dimensions?  
                    
            # Total loss
            loss = loss_kae

            # Update           
            optimizer_kae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kae.parameters(), max_norm=1.0)
            optimizer_kae.step()
            running_loss = running_loss + loss.detach().cpu().item()
            grad_norm = compute_grad_norm(kae)
            wandb.log({"loss_kae": loss_kae.item(), 'grad_norm': grad_norm}, step=epoch)

    # Save the trained Koopman Autoencoder
    kae.save("saved_models/KAEs/kae_cartpole_mlp.pt")
    wandb.finish()
