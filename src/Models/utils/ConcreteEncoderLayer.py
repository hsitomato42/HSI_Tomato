import torch
from torch import nn
import torch.nn.functional as F
import math

class ConcreteEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, start_temp=2.5, min_temp=0.01, alpha=0.99991, headstart_idx=None):
        super().__init__()
        self.headstart_idx = headstart_idx
        
        # Initialize device using device manager if available
        if device is None:
            try:
                from utils.device_manager import get_pytorch_device
                self.device = get_pytorch_device()
            except ImportError:
                # Fallback to CUDA if available, otherwise CPU
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.temp = start_temp

        self.logits = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.zeros_(self.logits)

        # Example of initializing certain bands to have higher logits:
        new_logits = self.logits.clone().detach()
        constant = 1
        if self.headstart_idx is not None:
            for i, idx in enumerate(self.headstart_idx):
                new_logits[i, idx] += constant

        # Example “loop” that tries to raise certain bands
        for i in range(new_logits.shape[0]):
            for idx in range(new_logits.shape[1]//new_logits.shape[0]+1):
                if idx + i*(math.ceil(new_logits.shape[1]/new_logits.shape[0])) >= new_logits.shape[1]:
                    break
                new_logits[i, idx + i*(math.ceil(new_logits.shape[1]/new_logits.shape[0]))] += constant

        # Assign the modified logits back
        with torch.no_grad():
            self.logits.copy_(new_logits)

        print(self.logits[0])
        print(self.logits[1])
        print(self.logits[2])
        print(self.logits.detach().cpu().numpy())

        self.regularizer = lambda: 0

    def forward(self, X, train=True, X_mask=None, debug=False):
        """
        X is expected to be shape (batch_size, <something>, H, W, input_dim),
        based on your original usage. Then we do the matrix multiply to select bands.

        If your data is 4D (batch_size, channels, H, W), you’ll need to adapt
        this code or reshape X so that the last dimension is ‘input_dim’.
        """
        uniform = torch.rand(self.logits.shape, device=self.device).clamp(min=1e-7)
        gumbel = -torch.log(-torch.log(uniform)) * 0.5
        self.temp = max(self.temp * self.alpha, self.min_temp)
        for _ in range(3):
            if self.temp > 1.0:
                self.temp *= self.alpha

        noisy_logits = (self.logits + gumbel) / self.temp

        if X_mask is not None:
            X *= X_mask
            logits_mask = X_mask.int() ^ 1
            noisy_logits = noisy_logits.reshape(1, self.logits.shape[0], -1)
            noisy_logits = torch.add(noisy_logits, logits_mask, alpha=-1e7)

        samples = F.softmax(noisy_logits, dim=-1)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[1]).float()
        selection = samples if train else discrete_logits

        # The code below expects X shape: (batch, ?, height, width, input_dim)
        # Then does X.transpose(2,4), so presumably X is 5D: [batch, 1, H, W, input_dim], etc.
        # Adjust as needed to match your actual input shape.
        Y = torch.matmul(
            X.transpose(2, 4),  # (batch, 1, w, h, input_dim) -> (batch, 1, input_dim, w, h)
            selection.transpose(-1, -2)  # (output_dim, input_dim) => (input_dim, output_dim)
        ).transpose(2, 4)

        if debug:
            return X, selection

        return Y

    def get_gates(self, mode):
        return self.logits.detach().cpu().numpy()

    def clustered_mask(self, mask, n_clusters):
        import numpy as np
        mask = np.array(mask)
        constant = 1
        new_logits = 0.0001 * self.logits.clone().detach() - 0.025
        for cluster_idx in range(n_clusters):
            cluster_bands = np.where(mask == cluster_idx, True, False)
            new_logits[cluster_idx, cluster_bands] += constant
        return new_logits