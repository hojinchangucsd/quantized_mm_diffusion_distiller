import torch
from q_helpers import *

device = "cpu"
FM_PATH = f"./checkpoints/celeba/base_6/checkpoint.pt"
fm, img_size, n_timesteps, time_scale = \
    load_float_ckpt(FM_PATH, batch_size=1, device=device)
del fm
QM_PATH = f"./checkpoints/quantized/fx_8-step_celeba_u.p"
quantized_model = torch.jit.load(QM_PATH)

quantized_diffusion = get_diffusion(quantized_model, n_timesteps, time_scale, device)

imgs = get_samples(n_timesteps, img_size, device, quantized_diffusion)
write_imgs('./images/quantized_out.png', imgs)
