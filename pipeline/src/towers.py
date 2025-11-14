from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class ItemTower(nn.Module):
    """
    Inputs expected per item at train time:
      - text_emb:   [B, T]   (frozen encoder transformer output, e.g., 384)
      - brand_id:   [B]
      - color_id:   [B]
      - price_oneh: [B, P]   (one-hot price band)
    Output:
      - item_vec:   [B, D]   (L2-normalized)
    """
    def __init__(self, text_dim: int, n_brands: int, n_colors: int, price_bins: int, d: int = 128):
        super().__init__()
        self.brand_emb = nn.Embedding(n_brands or 1, 32)
        self.color_emb = nn.Embedding(n_colors or 1, 16)
        self.price_lin = nn.Linear(price_bins or 1, 16, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(text_dim + 32 + 16 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, text_emb, brand_id, color_id, price_oneh):
        x = torch.cat([text_emb,
                       self.brand_emb(brand_id),
                       self.color_emb(color_id),
                       self.price_lin(price_oneh)], dim=-1)
        z = self.mlp(x)
        return F.normalize(z, p=2, dim=-1)
