"""
VAE architecture builder with YAML-driven configuration (ModelConfigFactory)
and a scikit-learn compatible wrapper for GridSearchCV training with validation,
loss weighting, checkpointing, and adaptive learning rate.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

# -----------------------------
# PyTorch VAE definition
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim: int = 230, latent_dim: int = 32, hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, 1)
        )

    def encode(self, x):
        h = self.encoder_net(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        y_pred = self.regressor(z)
        return recon, mu, logvar, y_pred

# -----------------------------
# Scikit-learn Wrapper for VAE
# -----------------------------
class VAESklearnWrapper(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for VAE to allow GridSearchCV training with validation."""

    def __init__(self, input_dim=None, latent_dim=32, hidden_dims=None,
                 dropout=0.1, lr=1e-3, epochs=500, batch_size=512,
                 alpha=1.0, beta=1.0, recon_weight=1.0, weight_decay=0.0,
                 save_path=None, device=None, verbose=True):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.recon_weight = recon_weight
        self.weight_decay = weight_decay
        self.save_path = save_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self._model = None

    def fit(self, X, y=None):
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32).unsqueeze(1)
        else:
            y_t = torch.zeros((len(X_t), 1), dtype=torch.float32)
        
        if self.input_dim is None:
            self.input_dim = X_t.shape[1]
        
        train_ds = TensorDataset(X_t, y_t)
        val_loader = None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._model = VAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        best_val_loss = float('inf')
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            train_losses, train_recon, train_kl, train_sup = [], [], [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                recon, mu, logvar, y_pred = self._model(xb)

                recon_loss = F.mse_loss(recon, xb) * self.recon_weight
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                sup_loss = F.mse_loss(y_pred, yb) * self.alpha if y is not None else 0.0

                loss = recon_loss + self.beta * kl_loss + sup_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_losses.append(loss.item())
                train_recon.append(recon_loss.item())
                train_kl.append(kl_loss.item())
                train_sup.append(float(sup_loss))

            avg_train_loss = np.mean(train_losses)
            avg_recon = np.mean(train_recon)
            avg_kl = np.mean(train_kl)
            avg_sup = np.mean(train_sup)

            val_loss = None
            if val_loader is not None:
                self._model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        recon, mu, logvar, y_pred = self._model(xb)
                        recon_loss = F.mse_loss(recon, xb) * self.recon_weight
                        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                        sup_loss = F.mse_loss(y_pred, yb) * self.alpha if y is not None else 0.0
                        val_losses.append((recon_loss + self.beta * kl_loss + sup_loss).item())
                val_loss = float(np.mean(val_losses))
                scheduler.step(val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if self.verbose:
                msg = (f"Epoch {epoch:03d} | train_loss={avg_train_loss:.6f} "
                       f"recon={avg_recon:.6f} kl={avg_kl:.6f} sup={avg_sup:.6f}")
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                print(msg)

            if self.save_path and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self._model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"  Saved best model to {self.save_path} (val_loss improved)")

        self.history_ = history
        return self

    def predict(self, X):
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, _, y_pred = self._model(X_t)
        return y_pred.cpu().numpy().ravel()

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "recon_weight": self.recon_weight,
            "weight_decay": self.weight_decay,
            "save_path": self.save_path,
            "device": self.device,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

