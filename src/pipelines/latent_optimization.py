import lpips
import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from tqdm import trange, tqdm
from pathlib import Path
from typing import Literal

from src.utils.weights import load_base_generator
from src.utils.alignment import FFHQFaceAligner
from src.utils.inversion import get_preprocess_transforms, save_image, save_latent
from src.losses.id_loss import IDLoss
from src.losses.w_reg_loss import WRegLoss


class LatentOptimizationPipeline:
    def __init__(
        self,
        resolution: Literal[256, 512, 1024] = 256,
        weights_dir: Path | str = 'weights',
        device: torch.device | Literal['cuda', 'cpu'] | str = 'cuda',
        w_avg_dir: Path | str = 'data',
        output_dir: Path | str = 'output/inversion_bank',
        lpips_net: Literal['alex', 'vgg'] = 'vgg',
        lr: float = 0.1,
        gamma: float = 0.99,
        lambda_l1: float = 1.0,
        lambda_l2: float = 0.1,
        lambda_lpips: float = 0.8,
        lambda_id: float = 0.5,
        lambda_reg: float = 0.01,
        save_every_n: int | None = 50,
        loging_every_n: int = 10,
        verbose: Literal[0, 1] = 1
    ) -> None:
        self.device = torch.device(device)
        weights_dir = Path(weights_dir)
        w_avg_dir = Path(w_avg_dir)
        self.output_dir = Path(output_dir)
        
        self.lr = lr
        self.gamma = gamma
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_lpips = lambda_lpips
        self.lambda_id = lambda_id
        self.lambda_reg = lambda_reg
        
        self.save_every_n = save_every_n
        self.loging_every_n = loging_every_n
        self.verbose = verbose
        
        self.generator = load_base_generator(resolution, weights_dir, device)
        self.generator = self.generator.eval().to(device)
        
        for param in self.generator.parameters():
            param.requires_grad = False
            
        self.aligner = FFHQFaceAligner(weights_dir)
        
        self.lpips_loss = lpips.LPIPS(net=lpips_net).to(device)
        self.id_loss = IDLoss(weights_dir, device)
        self.reg_loss = WRegLoss()
        
        self.w_avg = torch.load(w_avg_dir / f'w_avg_{resolution}.pt', map_location=device)
        
        self.transform = get_preprocess_transforms(resolution)
        
    def _format_log_message(
        self,
        step: int,
        steps: int,
        steps_len: int,
        total_loss: float,
        l1_loss: float,
        l2_loss: float,
        lpips_loss: float,
        id_loss: float,
        reg_loss: float,
        sep: str = ' | '
    ) -> str:
        message = (
            f'Step [{step:0{steps_len}d}/{steps}]{sep}'
            f'Total Loss: {total_loss:.4f}{sep}'
            f'L1 Loss: {l1_loss:.4f}{sep}'
            f'L2 Loss: {l2_loss:.4f}{sep}'
            f'LPIPS Loss: {lpips_loss:.4f}{sep}'
            f'ID Loss: {id_loss:.4f}{sep}'
            f'Reg Loss: {reg_loss:.4f}'
        )
        
        return message
        
    def __call__(
        self,
        image_path: Path | str,
        person_name: str,
        steps: int = 200,
        w_init: torch.Tensor | None = None,
        initial_noisy_steps: int = 0,
        noise_alpha: float = 0.05
    ) -> torch.Tensor:
        aligned_image = self.aligner(image_path)
        img_target = self.transform(aligned_image).unsqueeze(0).to(self.device)
        
        if w_init is not None:
            if w_init.ndim == 2:
                w_init = w_init.unsqueeze(0)
            
            assert w_init.shape == self.w_avg.shape, (
                f'Invalid shape for w_init. '
                f'Expected: {self.w_avg.shape}, got: {w_init.shape}'
            )
            
            w_opt = w_init.detach().clone().to(self.device).requires_grad_(True)
        
        else:
            w_opt = self.w_avg.clone().requires_grad_(True)
        
        optimizer = Adam([w_opt], lr=self.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        
        steps_len = len(str(steps))
        for step in trange(1, steps + 1, desc='Latent Optimization'):
            optimizer.zero_grad()
            
            if initial_noisy_steps > 0 and step <= initial_noisy_steps:
                noise_scale = noise_alpha * (1.0 - (step - 1) / initial_noisy_steps)
                w_noise = torch.randn_like(w_opt) * noise_scale
                current_w = w_opt + w_noise
            else:
                current_w = w_opt
            
            img_gen = self.generator.synthesis(current_w, noise_mode='const')
            
            l1_loss = F.l1_loss(img_gen, img_target)
            l2_loss = F.mse_loss(img_gen, img_target)
            
            pixel_loss = self.lambda_l1 * l1_loss + self.lambda_l2 * l2_loss
            lpips_loss = self.lpips_loss(img_gen, img_target).mean()
            id_loss = self.id_loss(img_gen, img_target)
            reg_loss = self.reg_loss(w_opt, self.w_avg)
            
            total_loss = (
                pixel_loss +
                self.lambda_lpips * lpips_loss +
                self.lambda_id * id_loss +
                self.lambda_reg * reg_loss
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if self.verbose and (step % self.loging_every_n == 0 or step == 1):
                message = self._format_log_message(
                    step, steps, steps_len, total_loss.item(), l1_loss.item(),
                    l2_loss.item(), lpips_loss.item(), id_loss.item(), reg_loss.item()
                )
                
                tqdm.write(message)
            
            if (self.save_every_n is not None) and (step % self.save_every_n == 0):
                save_image(w_opt, self.generator, self.output_dir / person_name / f'image_{step}.png')
                save_latent(w_opt, self.output_dir / person_name / f'w_{step}.pt')
        
        save_image(w_opt, self.generator, self.output_dir / person_name / f'image.png')
        save_latent(w_opt, self.output_dir / person_name / f'w.pt')
        
        return w_opt.detach()