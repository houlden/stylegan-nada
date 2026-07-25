import torch
import torch.nn as nn
from torchvision.transforms import transforms
from pathlib import Path
from typing import Literal, Callable

from src.utils.fix_random import seed_everything
from src.utils.image import tensor_to_pil


def get_preprocess_transforms(resolution: Literal[256, 512, 1024]) -> Callable:
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def compute_w_avg(
    generator: nn.Module,
    num_samples: int = 10_000,
    save_dir: Path | str = 'data',
    seed: int = 101
) -> Path:
    device = next(generator.parameters()).device
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f'w_avg_{generator.img_resolution}.pt'
    
    if not save_path.exists():
        seed_everything(seed)
        with torch.inference_mode():
            z_samples = torch.randn([num_samples, generator.z_dim], device=device)
            w_samples = generator.mapping(z_samples, c=None)
            w_avg = torch.mean(w_samples, dim=0, keepdim=True)
        
        torch.save(w_avg.cpu(), save_path)
        print(f'W_avg saved to: {save_path}')
    
    else:
        print(f'W_avg already exists at: {save_path}')
    
    return save_path

@torch.inference_mode()
def save_image(w: torch.Tensor, generator: nn.Module, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    img_tensor = generator.synthesis(w)
    img = tensor_to_pil(img_tensor)
    
    img.save(save_path)

def save_latent(w: torch.Tensor, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(w.detach().cpu(), save_path)


if __name__ == '__main__':
    import gc
    from src.utils.weights import load_base_generator
    
    for res in (256, 512, 1024):
        generator = load_base_generator(resolution=res)
        compute_w_avg(generator)
        
        del generator
        gc.collect()
        torch.cuda.empty_cache()