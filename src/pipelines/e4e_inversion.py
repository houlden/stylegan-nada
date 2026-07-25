import torch
from pathlib import Path
from typing import Literal

from src.utils.alignment import FFHQFaceAligner
from src.utils.inversion import get_preprocess_transforms
from src.models.e4e_invertor import E4EInvertor


class E4EInversionPipeline:
    def __init__(
        self,
        weights_dir: Path | str = 'weights',
        device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda'
    ) -> None:
        self.device = torch.device(device)
        
        self.aligner = FFHQFaceAligner(weights_dir)
        self.invertor = E4EInvertor(device)
        
        self.transform = get_preprocess_transforms(256)
    
    @torch.inference_mode()
    def __call__(self, image_path: Path | str) -> torch.Tensor:
        aligned_image = self.aligner(image_path)
        img_tensor = self.transform(aligned_image).unsqueeze(0).to(self.device)
        w_inverted = self.invertor(img_tensor)
        
        return w_inverted