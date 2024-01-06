import os
os.chdir("../")

device = "cpu"

def print_size(path): 
    print(f"Size: {os.path.getsize(path)/1e6:.2f} MB")

print_size("./checkpoints/quantized/fx_8-step_celeba_u.p")
print_size("./checkpoints/celeba/base_6/checkpoint.pt")
