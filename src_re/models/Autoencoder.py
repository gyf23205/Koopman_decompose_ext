import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim): 
        super(Encoder, self).__init__()

        print('Tanh ver')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim=None, hidden_dim=None, load=False, load_path=None):
        super(KoopmanAutoencoder, self).__init__()
        if load:
            self.load(load_path)
        else:
            assert state_dim is not None and hidden_dim is not None, "State and hidden dimensions must be provided if not loading."
            self.encoder = Encoder(state_dim, hidden_dim)
            self.decoder = Decoder(hidden_dim, state_dim)
            self.state_dim = state_dim
            self.hidden_dim = hidden_dim
            # self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
            self.K = torch.randn(hidden_dim, hidden_dim)  
    
    def forward(self, x):

        z = self.encoder(x)  
        
        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        x_hat = self.decoder(z)  
        return x_hat, z, z_next
        
    def compute_koopman_operator(self, latent_X, latent_Y):
        X_pseudo_inv = torch.linalg.pinv(latent_X)  # Compute pseudo-inverse of latent_X
        # # ###### REPLACE PINV
        # U, S, Vh = torch.linalg.svd(latent_X, full_matrices=False, driver='gesvda')
        # S_inv = 1.0 / S
        # X_pseudo_inv = Vh.T @ torch.diag(S_inv) @ U.T
        # ####################################
        self.K = torch.matmul(latent_Y.T, X_pseudo_inv.T)  # K = Y * X^+

    def save(self, path):
        save_dict = {
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'K': self.K
        }
        torch.save(save_dict, path)

    def load(self, path):
        load_dict = torch.load(path)
        self.state_dim = load_dict['state_dim']
        self.hidden_dim = load_dict['hidden_dim']
        self.encoder = Encoder(self.state_dim, self.hidden_dim)
        self.decoder = Decoder(self.hidden_dim, self.state_dim)
        self.encoder.load_state_dict(load_dict['encoder_state_dict'])
        self.decoder.load_state_dict(load_dict['decoder_state_dict'])
        self.K = load_dict['K']