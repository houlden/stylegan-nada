import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Literal

from src.utils.clip_utils import MEAN, STD, preprocess_for_clip, load_and_freeze_clip

class DirectionalCLIPLoss(nn.Module):
    def __init__(
        self,
        source_text: str,
        target_text: str,
        device: torch.device | Literal['cuda', 'cpu'] | str,
        clip_model: nn.Module | Literal[
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ] = 'ViT-B/32'
    ) -> None:
        super().__init__()
        
        if isinstance(clip_model, str):
            self.clip_model, _ = load_and_freeze_clip(clip_model, device)
        
        elif isinstance(clip_model, nn.Module):
            self.clip_model = clip_model
        
        else:
            raise TypeError('clip_model must be a str (model name) or an nn.Module object')
        
        self.register_buffer('mean', torch.tensor(MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD, device=device).view(1, 3, 1, 1))
        
        with torch.no_grad():
            tokenized_source = clip.tokenize([source_text]).to(device)
            tokenized_target = clip.tokenize([target_text]).to(device)
            
            text_feat_source = self.clip_model.encode_text(tokenized_source)
            text_feat_target = self.clip_model.encode_text(tokenized_target)
            
            text_feat_source /= text_feat_source.norm(dim=-1, keepdim=True)
            text_feat_target /= text_feat_target.norm(dim=-1, keepdim=True)
            
            direction = text_feat_target - text_feat_source
            direction /= direction.norm(dim=-1, keepdim=True)
            self.register_buffer('text_direction', direction)
    
    def forward(self, img_source: torch.Tensor, img_target: torch.Tensor) -> torch.Tensor:
        img_source_prep = preprocess_for_clip(img_source, self.mean, self.std)
        img_target_prep = preprocess_for_clip(img_target, self.mean, self.std)
        
        img_feat_source = self.clip_model.encode_image(img_source_prep)
        img_feat_target = self.clip_model.encode_image(img_target_prep)
        
        img_feat_source = img_feat_source / img_feat_source.norm(dim=-1, keepdim=True)
        img_feat_target = img_feat_target / img_feat_target.norm(dim=-1, keepdim=True)
        
        img_direction = img_feat_target - img_feat_source
        img_direction = img_direction / img_direction.norm(dim=-1, keepdim=True)
        
        loss = 1.0 - F.cosine_similarity(img_direction, self.text_direction, dim=-1)
        
        return loss.mean()
        
class GlobalCLIPLoss(nn.Module):
    def __init__(
        self,
        device: torch.device | Literal['cuda', 'cpu'] | str,
        clip_model: nn.Module | Literal[
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ] = 'ViT-B/32'
    ) -> None:
        super().__init__()
        
        if isinstance(clip_model, str):
            self.clip_model, _ = load_and_freeze_clip(clip_model, device)
        
        elif isinstance(clip_model, nn.Module):
            self.clip_model = clip_model
        
        else:
            raise TypeError('clip_model must be a string (model name) or an nn.Module object')
        
        self.register_buffer('mean', torch.tensor(MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD, device=device).view(1, 3, 1, 1))
    
    def forward(self, img: torch.Tensor, text: str) -> torch.Tensor:
        device = img.device
        
        img_prep = preprocess_for_clip(img, self.mean, self.std)
        
        tokenized_text = clip.tokenize([text]).to(device)
        
        img_feat = self.clip_model.encode_image(img_prep)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        text_feat = self.clip_model.encode_text(tokenized_text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        loss = 1.0 - F.cosine_similarity(img_feat, text_feat, dim=-1)
        
        return loss.mean()