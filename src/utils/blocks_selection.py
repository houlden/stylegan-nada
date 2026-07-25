import torch
import torch.nn as nn
from tqdm import trange
from typing import Literal

from src.models.stylegan_nada import StyleGANNADA
from src.losses.clip_losses import GlobalCLIPLoss

def select_frozen_blocks(
    model: StyleGANNADA,
    clip_model: nn.Module,
    target_text: str,
    k_trainable_blocks: int = 3,
    batch_size: int = 16,
    num_steps: int = 50,
    lr: float = 0.01,
    truncation_psi: float = 0.7,
    select_criterion: Literal['absolute', 'relative'] = 'absolute',
    norm: Literal['l1', 'l2'] = 'l2',
    disable_tqdm: bool = False,
    leave_tqdm: bool = False
) -> tuple[str, ...]:
    device = next(model.G_target.parameters()).device
    p_norm = 1 if (norm == 'l1') else 2
    
    z = torch.randn([batch_size, model.G_target.z_dim], device=device)
    
    with torch.no_grad():
        w_base = model.G_target.mapping(z, c=None, truncation_psi=truncation_psi)
    
    w_opt = w_base.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([w_opt], lr=lr)
    criterion = GlobalCLIPLoss(device=device, clip_model=clip_model)
    
    for _ in trange(
        num_steps,
        desc='Block selection',
        disable=disable_tqdm,
        leave=leave_tqdm
    ):
        img = model.G_target.synthesis(w_opt)
        loss = criterion(img, target_text)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    delta_w = w_opt.detach() - w_base
    
    if select_criterion == 'absolute':
        w_shifts = delta_w.norm(p=p_norm, dim=-1).mean(dim=0)
    
    elif select_criterion == 'relative':
        numerator = delta_w.norm(p=p_norm, dim=-1)
        denominator = w_base.norm(p=p_norm, dim=-1)
        w_shifts = (numerator / (denominator + 1e-8)).mean(dim=0)
    
    block_shifts = {}
    w_idx = 0
    
    for res in model.G_target.synthesis.block_resolutions:
        is_res_4 = (res == 4)
        length = 2 if is_res_4 else 3
        block_shifts[f'b{res}'] = w_shifts.narrow(0, w_idx, length).sum().item()
        w_idx += 1 if is_res_4 else 2

    selected_blocks = sorted(block_shifts, key=block_shifts.get, reverse=True)[:k_trainable_blocks]
    
    frozen_blocks = tuple(block for block in block_shifts if block not in selected_blocks)
    
    return frozen_blocks