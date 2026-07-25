import copy
import torch
import torch.nn as nn
from typing import Literal
from collections.abc import Collection

class StyleGANNADA(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        device: torch.device | Literal['cuda', 'cpu'] | str,
        frozen_blocks: Collection[str] = ('b4', 'b8', 'b16', 'b32')
    ) -> None:
        super().__init__()
        
        self.G_source = generator.eval().to(device)
        self.freeze_all_layers(self.G_source)
        
        self.G_target = copy.deepcopy(generator).train().to(device)
        self.setup_target_layers(frozen_blocks)
    
    @staticmethod
    def freeze_all_layers(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
    
    def setup_target_layers(self, frozen_blocks: Collection[str]) -> None:
        self.freeze_all_layers(self.G_target)
        
        for name, param in self.G_target.synthesis.named_parameters():
            is_frozen_block = any(f'{block}.' in name for block in frozen_blocks)
            is_affine_layer = '.affine.' in name
            is_torgb_layer = '.torgb.' in name
            
            is_trainable = not any([is_frozen_block, is_affine_layer, is_torgb_layer])

            if is_trainable:
                param.requires_grad = True
    
    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor | None = None,
        truncation_psi: float = 0.7
    ) -> tuple[torch.Tensor, torch.Tensor]:        
        img_source = self.G_source(z, c, truncation_psi=truncation_psi)
        img_target = self.G_target(z, c, truncation_psi=truncation_psi)
        return img_source, img_target