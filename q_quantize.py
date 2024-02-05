import torch
from torch.ao.quantization import (
    get_default_qconfig, get_default_qat_qconfig, float16_static_qconfig
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx, convert_fx, prepare_qat_fx
)
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
import q_train
from celeba_u import make_dataset as cu_make_dataset, make_model as make_celeba_u
from q_helpers import *
import unet_ddpm

CKPT_PATH = './checkpoints/celeba/base_6/checkpoint.pt'
device = torch.device("cuda:1")
backend = "x86"
batch_size = 1
lr = 1e-5
QAT = True
HALF = False # True for float16. False for int8

float_model, img_size, n_timesteps, time_scale \
    = load_float_ckpt(CKPT_PATH, batch_size, device)

qconfig = get_default_qconfig(backend) if not QAT else get_default_qat_qconfig(backend)
torch.backends.quantized.engine = backend
qconfig_mapping = QConfigMapping() \
    .set_global(qconfig) \
    .set_module_name("unet_ddpm.TimeEmbedding", None)
prepare_fx_custom_map = PrepareCustomConfig() \
    .set_non_traceable_module_classes([unet_ddpm.TimeEmbedding])

example_inputs = (torch.randn(img_size, device=device), 
                  torch.full((img_size[0],), 0, dtype=torch.int64).to(device))

if QAT: 
    float_model = float_model.train()

    num_iters = 100000
    
    train_args = q_train.make_argument_parser().parse_args([
        "--module", "celeba_u",
        "--name", "celeba",
        "--dname", f"{num_iters}iter_qat_8_step", 
        "--teacher_ckpt_path", f"./checkpoints/celeba/base_6/checkpoint.pt",
        "--num_timesteps", str(n_timesteps),
        "--time_scale", str(time_scale),
        "--batch_size", str(batch_size),
        "--num_iters", str(num_iters),
        "--lr", str(lr)
    ])

    prepared_model = prepare_qat_fx(float_model, qconfig_mapping, 
                                    example_inputs, prepare_fx_custom_map)

    def make_qp_model(): 
        model = prepared_model
        model.image_size = [1, 3, 256, 256]
        return model
    
    prepared_model = q_train.train_model(train_args, make_qp_model, 
                                         make_celeba_u, cu_make_dataset, device)

else: 
    float_model = float_model.eval()

    prepared_model = prepare_fx(float_model, qconfig_mapping, 
                                example_inputs, prepare_fx_custom_map)

    prepared_diffusion = get_diffusion(prepared_model, n_timesteps, time_scale, device)

    def calibrate(it=100): 
        with torch.no_grad():
            for _ in range(it): 
                get_samples(n_timesteps, img_size, device, prepared_diffusion)

    calibrate()

device = "cpu" if not HALF else device
prepared_model.eval().to(device)
quantized_model = convert_fx(prepared_model)
quantized_model.to(device)
train_str = "aware" if QAT else "post"
half_str = "float16" if HALF else "int8"

torch.jit.save(torch.jit.script(quantized_model,example_inputs=example_inputs), 
            f"./checkpoints/quantized/{half_str}_{train_str}_{n_timesteps}-step_celeba_u.p")
