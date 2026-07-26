import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
from pathlib import Path
from typing import Literal

def create_fixed_validation_set(
    num_samples: int = 4,
    z_dim: int = 512,
    save_dir: Path | str = 'data',
    seed: int = 101
) -> Path:
    save_path = Path(save_dir) / 'fixed_val_set.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not save_path.exists():
        torch.manual_seed(seed)
        fixed_z = torch.randn([num_samples, z_dim], device='cpu')
        torch.save(fixed_z, save_path)
        print(f'Fixed validation set saved to: {save_path}')
    else:
        print(f'Validation set already exists at: {save_path}')
    
    return save_path

@torch.no_grad()
def validate(
    model: nn.Module,
    z_val: torch.Tensor,
    step: int,
    images_dir: Path | str,
    truncation_psi: float = 0.7,
    save_mode: Literal['separated', 'combined', 'both'] = 'both'
) -> None:
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    model.G_target.eval()
    
    img_source = model.G_source(z_val, c=None, truncation_psi=truncation_psi, noise_mode='const')
    img_target = model.G_target(z_val, c=None, truncation_psi=truncation_psi, noise_mode='const')
    
    num_samples = z_val.shape[0]
    nrow = math.ceil(num_samples ** 0.5)
    
    grid_source = vutils.make_grid(img_source, nrow=nrow, padding=4,
                                   normalize=True, value_range=(-1, 1))
    grid_target = vutils.make_grid(img_target, nrow=nrow, padding=4,
                                   normalize=True, value_range=(-1, 1))
    
    if save_mode in ['separated', 'both']:
        vutils.save_image(grid_target, images_dir / f'target_{step}.png')
        if step == 1:
            vutils.save_image(grid_source, images_dir / f'source.png')
    
    if save_mode in ['combined', 'both']:
        comparison = torch.cat([grid_source, grid_target], dim=2)
        vutils.save_image(comparison, images_dir / f'comparison_{step}.png')
        
    model.G_target.train()


if __name__ == '__main__':
    create_fixed_validation_set()