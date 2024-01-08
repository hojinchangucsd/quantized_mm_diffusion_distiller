import torch
import cv2
import numpy as np
import argparse
from copy import deepcopy
from celeba_u import make_model as celeba_u
from v_diffusion import GaussianDiffusion, make_beta_schedule
from train_utils import p_sample_loop

def make_sampling_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", help="Index image generation should start from", type=int, required=True)
    parser.add_argument("--num_samples", help="Number of images to generate", type=int, required=True)
    parser.add_argument("--out_folder", help="Path to put images", type=str, required=True)
    parser.add_argument("--device", help="Device to run sampling on", type=str, required=True)
    return parser

def get_samples(num_samples, image_size, device, diffusion): 
    noise = torch.randn(image_size, device=device)
    return p_sample_loop(diffusion, noise, {}, device, samples_to_capture=num_samples, clip_value=1.2)

def write_imgs(path, imgs): 
    images_ = []
    for images in imgs:
        images = images.split(1, dim=0)
        images = torch.cat(images, -1)
        images_.append(images)
    images_ = torch.cat(images_, 2)
    images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    img = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(path, img)

def load_float_ckpt(path, batch_size, device): 
    float_model = celeba_u().to(device)
    ckpt = torch.load(path)
    float_model.load_state_dict(ckpt["G"])
    float_model.eval()
    img_size = deepcopy(float_model.image_size)
    img_size[0] = batch_size
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    return float_model, img_size, n_timesteps, time_scale

def get_diffusion(model, n_timesteps, time_scale, device): 
    betas = make_beta_schedule("cosine", n_timesteps, cosine_s=8e-3).to(device)
    return GaussianDiffusion(model, betas, time_scale)

