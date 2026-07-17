import bz2
import gdown
import torch
import torch.nn as nn
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Literal

from third_party.stylegan2 import legacy

# StyleGAN weights link
FFHQ_256_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl'
FFHQ_512_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl'
FFHQ_1024_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl'

# E4E weights ID
FFHQ_E4E_ID = '1cUv_reLE6k3604or78EranS7XzuVMWeO'

# Shape Predictor weights archive link
SHAPE_PREDICTOR_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

resolution_dict = {
    256: FFHQ_256_URL,
    512: FFHQ_512_URL,
    1024: FFHQ_1024_URL
}

def download_stylegan_weights(
    resolution: Literal[256, 512, 1024] = 256,
    weights_dir: Path | str = 'weights'
) -> Path:
    assert resolution in resolution_dict, (
        f'Incorrect resolution: {resolution}. '
        f'Select one of: {list(resolution_dict.keys())}'
    )
    
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    url = resolution_dict[resolution]
    filename = Path(urlparse(url).path).name
    local_path = weights_dir / filename
    
    if not local_path.exists():
        print('Loading StyleGAN weights to a local drive...')
        urllib.request.urlretrieve(url=url, filename=local_path)
        print(f'StyleGAN weights are uploaded to: {local_path}.')
    else:
        print(f'StyleGAN weights are uploaded to: {local_path}.')
    
    return local_path

def download_e4e_weights(
    gdown_id: str = FFHQ_E4E_ID,
    weights_dir: Path | str = 'weights',
    filename: str = 'e4e_ffhq_encode.pt'
) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = weights_dir / filename
    
    if not local_path.exists():
        print('Loading E4E weights to a local drive...')
        gdown.download(id=gdown_id, output=str(local_path))
        print(f'E4E weights are uploaded to: {local_path}.')
    else:
        print(f'E4E weights are uploaded to: {local_path}.')
    
    return local_path

def unpack_bz2_archive(
    input_path: Path | str,
    output_path: Path | str,
    delete_archive: bool = True
) -> None:
    input_path, output_path = Path(input_path), Path(output_path)
    
    assert input_path.exists(), 'input_path does not exist'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with bz2.open(input_path, 'rb') as inp, open(output_path, 'wb') as outp:
        outp.write(inp.read())
    
    if delete_archive:
        input_path.unlink()

def download_shape_predictor_weights(
    url: str = SHAPE_PREDICTOR_URL,
    weights_dir: Path | str = 'weights',
    delete_archive: bool = True
) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(urlparse(url).path).name
    archive_path = weights_dir / filename
    output_path = archive_path.with_suffix('')
    
    if output_path.exists():
        print(f'Shape Predictor weights are uploaded to: {output_path}.')
        return output_path
    
    if not archive_path.exists():
        print('Loading Shape Predictor weights to a local drive...')
        urllib.request.urlretrieve(url=url, filename=archive_path)

    print('Unpacking Shape Predictor weights...')
    unpack_bz2_archive(archive_path, output_path, delete_archive)
    print(f'Shape Predictor weights are unpacked to: {output_path}.')
    
    return output_path

def load_base_generator(path: Path | str) -> nn.Module:
    with open(path, 'rb') as f:
        network_dict = legacy.load_network_pkl(f)
        return network_dict['G_ema']

def save_style_weights(
    generator: nn.Module,
    save_path: Path | str,
    save_requires_grad_True_only: bool = False,
    verbose: Literal[0, 1] = 1
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_requires_grad_True_only:
        style_state_dict = {
            name: param.detach().cpu()
            for name, param in generator.synthesis.named_parameters()
            if param.requires_grad
        }
    else:
        style_state_dict = generator.synthesis.state_dict()
    
    torch.save(style_state_dict, save_path)
    
    if verbose:
        print(f'Style weights saved in: {save_path}')

def load_style_weights(
    generator: nn.Module,
    weights_path: Path | str
) -> nn.Module:
    device = next(generator.parameters()).device
    
    weights_path = Path(weights_path)
    assert weights_path.exists(), f'File not found in: {weights_path}'
    
    style_state_dict = torch.load(weights_path, map_location=device)
    generator.synthesis.load_state_dict(style_state_dict, strict=False)
    
    return generator

def load_styled_generator(
    style_weights_path: Path | str,
    resolution: Literal[256, 512, 1024] = 256,
    base_weights_dir: Path | str = 'weights',
    device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda'
) -> nn.Module:
    device = torch.device(device)
    
    base_weights_path = download_stylegan_weights(resolution, base_weights_dir)
    generator = load_base_generator(base_weights_path).to(device)
    generator = load_style_weights(generator, style_weights_path)
    generator.eval()
    
    return generator