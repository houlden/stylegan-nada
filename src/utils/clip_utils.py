import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Literal, Callable


MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_for_clip(
    img_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    img = torch.clamp((img_tensor + 1.0) / 2.0, 0.0, 1.0)
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    img = (img - mean) / std
    return img

def load_and_freeze_clip(
    model_name: Literal[
        'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    ] = 'ViT-B/32',
    device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda'
) -> tuple[nn.Module, Callable]:
    device = torch.device(device)
    
    model, preprocess = clip.load(model_name, device=device)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model, preprocess