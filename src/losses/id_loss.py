import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal

from third_party.arcface.model_irse import Backbone
from src.utils.weights import download_irse50_weights


class IDLoss(nn.Module):
    def __init__(
        self,
        weights_dir: Path | str = 'weights',
        device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda'
    ) -> None:
        super().__init__()
        
        weights_path = download_irse50_weights(weights_dir=weights_dir)
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.facenet.to(device)
        self.facenet.eval()
        
        for param in self.facenet.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
    
    def extract_feats(self, x):
        x = torch.clamp(x, -1.0, 1.0)
        
        if x.shape[2] != 256:
            x = self.pool(x)
            
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        
        return x_feats

    def forward(self, y_hat, y):
        y_feat = self.extract_feats(y).detach()
        y_hat_feat = self.extract_feats(y_hat)
        
        similarity = torch.sum(y_hat_feat * y_feat, dim=-1)
        loss = 1.0 - torch.mean(similarity)
        
        return loss