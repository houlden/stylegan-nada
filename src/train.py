import torch
import argparse
from pathlib import Path
from typing import Literal
from collections.abc import Collection
from tqdm import trange, tqdm

from src.models.stylegan_nada import StyleGANNADA
from src.losses.clip_losses import DirectionalCLIPLoss
from src.utils.weights import load_base_generator, save_style_weights
from src.utils.fix_random import seed_everything
from src.utils.validation import validate
from src.utils.clip_utils import load_and_freeze_clip
from src.utils.blocks_selection import select_frozen_blocks
from src.utils.training_utils import warmup_generator, format_log_message, save_experiment_config


def train(
    # Basic
    source_text: str = 'a photo of a person',
    target_text: str = 'a sketch of a person',
    experiment_name: str = 'sketch',
    resolution: Literal[256, 512, 1024] = 256,
    device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda',
    seed: int | None = None,
    # Hyperparameters NADA
    num_steps: int = 300,
    batch_size: int = 4,
    lr: float = 0.002,
    truncation_psi: float = 0.7,
    # CLIP model
    clip_model_name: Literal[
        'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    ] = 'ViT-B/32',
    # Blocks selection parameters
    blocks_selection_mode: Literal['static', 'once', 'adaptive'] = 'static',
    blocks_to_freeze: Collection[str] = ('b4', 'b8', 'b16', 'b32'),
    k_trainable_blocks: int = 3,
    select_batch_size: int = 16,
    select_num_steps: int = 50,
    select_lr: float = 0.01,
    select_criterion: Literal['absolute', 'relative'] = 'absolute',
    select_norm: Literal['l1', 'l2'] = 'l2',
    adaptive_selection_every_n: int = 50,
    # Logging, validation, checkpoints
    weights_dir: Path | str = 'weights',
    output_dir: Path | str = 'output/styles',
    loging_every_n: int = 10,
    save_weights_every_n: int | None = 50,
    validate_every_n: int | None = 50,
    val_set_path: Path | str | None = 'data/fixed_val_set.pt',
    verbose: Literal[0, 1, 2] = 2
) -> None:
    config_snapshot = locals().copy()
    
    device = torch.device(device)
    
    weights_dir = Path(weights_dir)
    experiment_dir = Path(output_dir) / f'{experiment_name}'
    experiment_dir.mkdir(parents=True, exist_ok=True)
    images_dir = experiment_dir / 'images'
    style_weights_dir = experiment_dir / 'weights'
    val_set_path = Path(val_set_path) if val_set_path is not None else None
    
    save_experiment_config(config_snapshot, experiment_dir / 'config.json')
    
    generator = load_base_generator(resolution, weights_dir, device)
    
    clip_model, _ = load_and_freeze_clip(clip_model_name, device)
    
    model = StyleGANNADA(generator, device=device, frozen_blocks=())
    
    if blocks_selection_mode == 'once':
        blocks_to_freeze = select_frozen_blocks(
            model=model,
            clip_model=clip_model,
            target_text=target_text,
            k_trainable_blocks=k_trainable_blocks,
            batch_size=select_batch_size,
            num_steps=select_num_steps,
            lr=select_lr,
            truncation_psi=truncation_psi,
            select_criterion=select_criterion,
            norm=select_norm,
            disable_tqdm=(verbose < 1),
            leave_tqdm=True
        )
        
        if verbose > 1:
                print(f'Frozen blocks: {", ".join(blocks_to_freeze)}')
    
    if blocks_selection_mode != 'adaptive':
        model.setup_target_layers(blocks_to_freeze)
        
    trainable_params = [p for p in model.G_target.synthesis.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=trainable_params, lr=lr, betas=(0.0, 0.99))
    
    criterion = DirectionalCLIPLoss(
        source_text=source_text,
        target_text=target_text,
        device=device,
        clip_model=clip_model
    )
    
    need_validation = all([
        validate_every_n is not None,
        val_set_path is not None,
        val_set_path.exists()
    ])
    
    if need_validation:
        z_val = torch.load(val_set_path, map_location=device)
    
    warmup_generator(model.G_source)
    
    num_steps_len = len(str(num_steps))
    for step in trange(1, num_steps + 1, desc='Training'):
        if (
            blocks_selection_mode == 'adaptive'
            and (step % adaptive_selection_every_n == 0 or step == 1)
            and step != num_steps
        ):
            blocks_to_freeze = select_frozen_blocks(
                model=model,
                clip_model=clip_model,
                target_text=target_text,
                k_trainable_blocks=k_trainable_blocks,
                batch_size=select_batch_size,
                num_steps=select_num_steps,
                lr=select_lr,
                truncation_psi=truncation_psi,
                select_criterion=select_criterion,
                norm=select_norm,
                disable_tqdm=(verbose < 1),
                leave_tqdm=False
            )
            
            model.setup_target_layers(blocks_to_freeze)
        
        z = torch.randn([batch_size, model.G_target.z_dim], device=device)
        img_source, img_target = model(z, truncation_psi=truncation_psi)
        
        loss = criterion(img_source, img_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step % loging_every_n == 0 or step == 1):
            message = format_log_message(
                verbose, step, num_steps, num_steps_len, loss.item(),
                blocks_selection_mode, blocks_to_freeze
            )
            
            tqdm.write(message)
        
        if need_validation and (step % validate_every_n == 0 or step == 1):
            validate(model, z_val, step, images_dir, truncation_psi=truncation_psi)
        
        if (save_weights_every_n is not None) and (step % save_weights_every_n == 0):
            save_style_weights(
                generator=model.G_target,
                save_path=style_weights_dir / f'{experiment_name}_{step}.pt',
                save_requires_grad_True_only=(blocks_selection_mode != 'adaptive'),
                verbose=0
            )
    
    save_style_weights(
        generator=model.G_target,
        save_path=style_weights_dir / f'{experiment_name}.pt',
        save_requires_grad_True_only=(blocks_selection_mode != 'adaptive')
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StyleGAN-NADA Training Script")
    
    # Basic
    parser.add_argument('--source_text', type=str, default='a photo of a person')
    parser.add_argument('--target_text', type=str, default='a sketch of a person')
    parser.add_argument('--experiment_name', type=str, default='sketch')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=101)
    # Hyperparameters NADA
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--truncation_psi', type=float, default=0.7)
    # CLIP model
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32')
    # Blocks selection parameters
    parser.add_argument('--blocks_selection_mode', type=str, default='static')
    parser.add_argument('--blocks_to_freeze', type=str, nargs='+', default=['b4', 'b8', 'b16', 'b32'])
    parser.add_argument('--k_trainable_blocks', type=int, default=3)
    parser.add_argument('--select_batch_size', type=int, default=16)
    parser.add_argument('--select_num_steps', type=int, default=50)
    parser.add_argument('--select_lr', type=float, default=0.01)
    parser.add_argument('--select_criterion', type=str, default='absolute')
    parser.add_argument('--select_norm', type=str, default='l2')
    parser.add_argument('--adaptive_selection_every_n', type=int, default=50)
    # Logging, validation, checkpoints
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--output_dir', type=str, default='output/styles')
    parser.add_argument('--loging_every_n', type=int, default=10)
    parser.add_argument('--save_weights_every_n', type=int, default=50)
    parser.add_argument('--validate_every_n', type=int, default=50)
    parser.add_argument('--val_set_path', type=str, default='data/fixed_val_set.pt')
    parser.add_argument('--verbose', type=int, default=2)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    seed_everything(seed=args.seed)
    
    args_dict = vars(args)
    
    train(**args_dict)