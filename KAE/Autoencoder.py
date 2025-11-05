import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim): 
        super(Encoder, self).__init__()

        print('Tanh ver')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*4),
            nn.Tanh(),
            nn.Linear(hidden_dim*4, hidden_dim*3),
            nn.Tanh(),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observable_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)
    
class Encoder_tf(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim, num_layers=3, num_heads=4, dropout=0.1):
        super(Encoder_tf, self).__init__()

        print('Transformer ver (official PyTorch)')

        # Input projection to match Transformer dimension
        self.input_linear = nn.Linear(state_dim, hidden_dim)

        # Official PyTorch TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',  # official default
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to observable_dim
        self.output_linear = nn.Linear(hidden_dim, observable_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, state_dim) or (batch_size, seq_len, state_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        # Project to hidden dim
        x = self.input_linear(x)  # (B, L, H)
        x = x.transpose(0, 1)     # (L, B, H) for Transformer

        # Pass through Transformer
        x = self.encoder(x)

        # Pool over sequence (mean)
        x = x.mean(dim=0)         # (B, H)

        # Final projection + activation
        return self.activation(self.output_linear(x))
    
class Encoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim): 
        super(Encoder_walk, self).__init__()


        print('Walking Tanh ver')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*4),
            nn.Tanh(),
            nn.Linear(hidden_dim*4, hidden_dim*3),
            nn.Tanh(),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observable_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, observable_dim, state_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(observable_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim,device):
        super(KoopmanAutoencoder, self).__init__()
        self.encoder = Encoder(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        # self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
        self.K = torch.randn(observable_dim, observable_dim).to(device)  
    
    def forward(self, x):

        z = self.encoder(x)  

        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)  
        return x_hat, z, y_hat
        
    def compute_koopman_operator(self, latent_X, latent_Y,device):
        latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]
        X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        self.K = latent_Y.T @ X_pseudo_inv

class KoopmanAutoencoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim,device):
        super(KoopmanAutoencoder_walk, self).__init__()
        self.encoder = Encoder_walk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        # self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
        self.K = torch.randn(observable_dim, observable_dim).to(device)  
    
    def forward(self, x):

        z = self.encoder(x)  

        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)  
        return x_hat, z, y_hat
    
        
    def compute_koopman_operator(self, latent_X, latent_Y, device):
        # # # print(latent_X.shape, latent_Y.shape)
        # latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        # latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]

        # damping=1e-4
        # eps=1e-8
        # A = latent_X.T
        # m, n = A.shape
        # if m >= n:
        #     # tall or square: (AᵀA + λI)^(-1) Aᵀ
        #     I = torch.eye(n, device=A.device, dtype=A.dtype)
        #     AtA = A.T @ A + damping * I
        #     X_pseudo_inv = torch.linalg.solve(AtA, A.T)
        # else:
        #     # wide: Aᵀ (A Aᵀ + λI)^(-1)
        #     I = torch.eye(m, device=A.device, dtype=A.dtype)
        #     AAt = A @ A.T + damping * I
        #     X_pseudo_inv = A.T @ torch.linalg.solve(AAt, I)
                
        # # X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        # self.K = latent_Y.T @ X_pseudo_inv

        # damping=1e-6
        # d = latent_X.size(1)
        # I = torch.eye(d, device=device, dtype=latent_X.dtype)

        # # single GEMM for both products
        # XY = torch.cat([latent_X, latent_Y], dim=1)           # [N, 2d]
        # XtXY = latent_X.T @ XY                                # [d, 2d]
        # XtX, XtY = XtXY[:, :d], XtXY[:, d:]                   # split results

        # K_T = torch.linalg.solve(XtX + damping * I, XtY)
        # self.K = K_T.T                         # [d, d]

        latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]
        X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        self.K = latent_Y.T @ X_pseudo_inv
        

class KoopmanAutoencoder_walk_tf(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim, num_layer, head_dim, device):
        super(KoopmanAutoencoder_walk_tf, self).__init__()
        self.encoder = Encoder_tf(state_dim, hidden_dim, observable_dim, num_layers=num_layer, num_heads = head_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        # self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
        self.K = torch.randn(observable_dim, observable_dim).to(device)  
    
    def forward(self, x):

        z = self.encoder(x)  

        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)  
        return x_hat, z, y_hat
    
        
    def compute_koopman_operator(self, latent_X, latent_Y, device):
        latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]
        X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        self.K = latent_Y.T @ X_pseudo_inv

class Original_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(235, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        return self.model(x)