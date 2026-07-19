import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def tensor_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.detach().cpu().permute(1, 2, 0)
    
    elif img_tensor.ndim == 4:
        img_tensor = img_tensor.detach().cpu().permute(0, 2, 3, 1)
    
    img_np = (img_tensor * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).numpy()
    
    return img_np


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    if img_tensor.ndim == 4:
        assert img_tensor.size(0) == 1, (
            f'Expected a single image or tensor with batch size: 1, '
            f'but got a tensor with batch size: {img_tensor.size(0)}'
        )
        
        img_tensor = img_tensor.squeeze(0)
    
    img_np = tensor_to_numpy(img_tensor)
    
    img_pil = Image.fromarray(img_np)
    
    return img_pil


def generate_one_image(generator: nn.Module, truncation_psi: float = 0.7) -> Image.Image:
    device = next(generator.parameters()).device
    
    z = torch.randn([1, generator.z_dim], device=device)
    
    with torch.inference_mode():
        img_tensor = generator(z, c=None, truncation_psi=truncation_psi)
    
    img_pil = tensor_to_pil(img_tensor)
    
    return img_pil