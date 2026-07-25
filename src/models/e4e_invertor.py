import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing import Literal

from src.utils.weights import download_e4e_weights
from third_party.e4e.psp_encoders import Encoder4Editing


class E4EInvertor(nn.Module):
    def __init__(self, device: torch.device | Literal['cuda', 'cpu'] | str) -> None:
        super().__init__()
        
        checkpoint_path = download_e4e_weights()
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        self.latent_avg = ckpt['latent_avg'].to(device)
        
        encoder_state_dict = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
        consume_prefix_in_state_dict_if_present(encoder_state_dict, prefix='encoder.')
        
        self.encoder = Encoder4Editing(num_layers=50, mode='ir_se')
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.encoder.to(device)
        self.encoder.eval()
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        w_plus = self.latent_avg.unsqueeze(0) + self.encoder(img_tensor)
        return w_plus