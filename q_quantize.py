import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from q_helpers import *

import unet_ddpm

CKPT_PATH = './checkpoints/celeba/base_6/checkpoint.pt'
device = torch.device("cuda:1")
batch_size = 1

float_model, img_size, n_timesteps, time_scale \
    = load_float_ckpt(CKPT_PATH, batch_size, device)

qconfig = get_default_qconfig("x86")
torch.backends.quantized.engine = 'x86'
qconfig_mapping = QConfigMapping().set_global(qconfig)
custom_q_map = PrepareCustomConfig() \
    .set_non_traceable_module_classes([unet_ddpm.TimeEmbedding])

example_inputs = (torch.randn(img_size, device=device), 
                  torch.full((img_size[0],), 0, dtype=torch.int64).to(device))

prepared_model = prepare_fx(float_model, qconfig_mapping, 
                            example_inputs, custom_q_map)

prepared_diffusion = get_diffusion(prepared_model, n_timesteps, time_scale, device)

def calibrate(it=100): 
    with torch.no_grad():
        for _ in range(it): 
            get_samples(n_timesteps, img_size, device, prepared_diffusion)

calibrate(10)

device = "cpu"
prepared_model.to(device)
quantized_model = convert_fx(prepared_model)
quantized_model.to(device)

torch.jit.save(torch.jit.script(quantized_model,example_inputs=example_inputs), 
               f"./checkpoints/quantized/fx_{n_timesteps}-step_celeba_u.p")
