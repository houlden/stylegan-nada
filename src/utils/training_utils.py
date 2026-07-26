import json
import torch
import torch.nn as nn
from typing import Literal, Any
from collections.abc import Collection
from pathlib import Path

def save_experiment_config(config_dict: dict[str, Any], save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    clean_config = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            clean_config[key] = value
        except (TypeError, OverflowError):
            clean_config[key] = str(value)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(clean_config, f, indent=4, ensure_ascii=False)

@torch.inference_mode()
def warmup_generator(generator: nn.Module) -> None:
    device = next(generator.parameters()).device
    
    warmup_z = torch.randn([1, generator.z_dim], device=device)
    _ = generator(warmup_z, c=None)

def format_log_message(
    verbose: Literal[0, 1, 2],
    step: int,
    num_steps: int,
    num_steps_len: int,
    loss: float,
    blocks_selection_mode: Literal['static', 'once', 'adaptive'],
    blocks_to_freeze: Collection[str],
    sep: str = ' | '
) -> str:
    message = (
        f'Step [{step:0{num_steps_len}d}/{num_steps}]{sep}'
        f'CLIP Directional Loss: {loss:.4f}{sep}'
    )
    
    if (blocks_selection_mode == 'adaptive') and (verbose > 1):
        message += f'Frozen blocks: {", ".join(blocks_to_freeze)}'
    
    return message