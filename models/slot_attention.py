import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=128, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Learned slot initialization parameters.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        
        # FIX 2: Initialize sigma to small positive values.
        self.slots_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_sigma)  # Ensures controlled variance.
        
        # Linear projections.
        self.project_inputs = nn.Linear(dim, dim)
        self.project_slots = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        # GRU for slot update.
        self.gru = nn.GRUCell(dim, dim)

        # MLP for further refinement (with hidden layer of size 128).
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # LayerNorm for stability.
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [B, N, D]
        Returns:
            slots: Tensor of shape [B, num_slots, D]
        """
        B, N, D = inputs.shape
        assert D == self.dim, "Input dimension must match slot dimension."
        inputs = self.norm_inputs(inputs)

        # Initialize slots using a learned Gaussian with controlled variance.
        slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.num_slots, self.dim, device=inputs.device)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # Compute queries from slots, keys and values from inputs.
            k = self.project_inputs(inputs)            # [B, N, D]
            v = self.project_v(inputs)                 # [B, N, D]
            q = self.project_slots(slots_norm)         # [B, num_slots, D]

            # Compute scaled dot-product attention logits.
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) / math.sqrt(D)
            
            # FIX 1: Apply softmax over the SLOTS dimension (dim=1) to enforce competition.
            attn = F.softmax(attn_logits, dim=1)  # [B, num_slots, N]
            
            # Compute weighted sum over values.
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B, num_slots, D]

            # Update slots using GRU and add residual MLP refinement.
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
