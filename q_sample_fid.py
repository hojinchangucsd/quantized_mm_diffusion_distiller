import torch
import os
from q_helpers import *

parser = make_sampling_argparser()
args = parser.parse_args()

device = "cpu"
FM_PATH = f"./checkpoints/celeba/base_6/checkpoint.pt"
_, img_size, n_timesteps, time_scale = \
    load_float_ckpt(FM_PATH, batch_size=1, device=device)
qm_path = f"./checkpoints/quantized/aware_fx_8-step_celeba_u.p"
quantized_model = torch.jit.load(qm_path)

quantized_diffusion = get_diffusion(quantized_model, n_timesteps, time_scale, device)

start = args.start_index
end = start + args.num_samples
for i in range(start,end): 
    img = get_samples(1, img_size, device, quantized_diffusion)
    os.makedirs(args.out_folder, exist_ok=True)
    img_path = os.path.join(args.out_folder, f'{i}.jpg')
    write_imgs(img_path, img)
