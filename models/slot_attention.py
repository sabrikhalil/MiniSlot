import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=7, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Learned slot initialization parameters.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        # Linear projections.
        self.project_inputs = nn.Linear(dim, dim)
        self.project_slots = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        # GRU for slot update.
        self.gru = nn.GRUCell(dim, dim)

        # MLP for further refinement.
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # LayerNorm for stability.
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [B, N, D] where N is the number of input features.
        Returns:
            slots: Tensor of shape [B, num_slots, D]
        """
        B, N, D = inputs.shape
        assert D == self.dim, "Input dimension must match slot dimension."

        # Normalize inputs.
        inputs = self.norm_inputs(inputs)

        # Initialize slots by sampling from a learned Gaussian.
        slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.num_slots, self.dim, device=inputs.device)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # Project inputs and slots.
            # keys and values: from inputs, shape [B, N, D]
            k = self.project_inputs(inputs)    # [B, N, D]
            v = self.project_v(inputs)         # [B, N, D]
            # queries: from slots, shape [B, num_slots, D]
            q = self.project_slots(slots_norm) # [B, num_slots, D]

            # Compute scaled dot-product attention.
            # We want each slot (query) to attend over all input features (keys).
            # Compute logits of shape [B, num_slots, N]:
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) / math.sqrt(D)
            # Apply softmax over the input features dimension (N) for each slot.
            attn = F.softmax(attn_logits, dim=-1)  # [B, num_slots, N]

            # Compute slot updates as the weighted sum over input values.
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B, num_slots, D]

            # Update slots using GRU: process each slot independently.
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)

            # Apply a residual MLP refinement.
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
