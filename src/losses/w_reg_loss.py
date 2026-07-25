import torch
import torch.nn as nn
import torch.nn.functional as F


class WRegLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, w_opt: torch.Tensor, w_avg: torch.Tensor = None) -> torch.Tensor:
        return F.mse_loss(w_opt, w_avg)