import torch
from torchvision.transforms import transforms
from pathlib import Path

from src.models.E4EInvertor import E4EInvertor
from src.utils.alignment import FFHQFaceAligner


preprocess_for_e4e = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def invert_one_image(
    image_path: Path | str,
    aligner: FFHQFaceAligner,
    invertor: E4EInvertor,
) -> torch.Tensor:
    device = next(invertor.parameters()).device
    
    aligned_image = aligner(image_path)
    
    img_tensor = preprocess_for_e4e(aligned_image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        w_inverted = invertor(img_tensor)
    
    return w_inverted
