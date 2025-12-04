# file: vae_lightning_module.py
import os
import tempfile
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LightningVAE(L.LightningModule):
    """
    Lightning VAE with regression head and optional categorical embeddings.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 lr: float = 1e-3,
                 n_categorical: int = 0,
                 embedding_sizes: Optional[List[int]] = None,
                 recon_weight: float = 1.0,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.save_hyperparameters()

        self.n_categorical = n_categorical
        self.embedding_sizes = embedding_sizes or []
        self.recon_weight = recon_weight
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

        if n_categorical != len(self.embedding_sizes):
            raise ValueError("Length of embedding_sizes must match n_categorical")

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList()
        for n_cat, emb_dim in zip(self.embedding_sizes, self.embedding_sizes):
            self.embeddings.append(nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim))

        # Adjust input_dim for embeddings
        total_input_dim = input_dim + sum(self.embedding_sizes)

        # Encoder
        if hidden_dims is None:
            hidden_dims = [128, 64]
        enc_layers = []
        in_dim = total_input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
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

    def _embed_categorical(self, x_cat):
        if x_cat is None or self.n_categorical == 0:
            return []
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(x_cat[:, i]))
        return embeds

    def forward(self, x_num, x_cat=None):
        x_cat_embed = self._embed_categorical(x_cat)
        x = torch.cat([x_num] + x_cat_embed, dim=1) if x_cat_embed else x_num
        h = self.encoder_net(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        y_pred = self.regressor(z)
        return recon, mu, logvar, y_pred

    def encode(self, x_num, x_cat=None):
        x_cat_embed = self._embed_categorical(x_cat)
        x = torch.cat([x_num] + x_cat_embed, dim=1) if x_cat_embed else x_num
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_net(z)

    def _loss(self, x, recon, mu, logvar, y_pred, y_true):
        recon_loss = F.mse_loss(recon, x, reduction='mean') * self.recon_weight
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        sup_loss = F.mse_loss(y_pred, y_true, reduction='mean') * self.alpha
        return recon_loss + self.beta * kld + sup_loss, recon_loss, kld, sup_loss

    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        recon, mu, logvar, y_pred = self(x_num, x_cat)
        loss, _, _, _ = self._loss(x_num, recon, mu, logvar, y_pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        recon, mu, logvar, y_pred = self(x_num, x_cat)
        loss, _, _, _ = self._loss(x_num, recon, mu, logvar, y_pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Utility function to prepare tensors
# -----------------------------
def prepare_data(X: pd.DataFrame, label_encoders: Optional[Dict[str, LabelEncoder]] = None):
    """
    Convert DataFrame to numeric and categorical tensors.
    Returns: (X_num: torch.FloatTensor, X_cat: torch.LongTensor or None)
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Detect categorical columns
    if label_encoders is None:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        label_encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in categorical_cols}
    else:
        categorical_cols = list(label_encoders.keys())

    X_cat = None
    if categorical_cols:
        X_cat = np.zeros((len(X), len(categorical_cols)), dtype=np.int64)
        for i, col in enumerate(categorical_cols):
            X_cat[:, i] = label_encoders[col].transform(X[col].astype(str))
        X_cat = torch.tensor(X_cat, dtype=torch.long)

    X_num = X.drop(columns=categorical_cols, errors='ignore').apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    X_num = torch.tensor(X_num, dtype=torch.float32)

    return X_num, X_cat, label_encoders
