import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from train_utils import *

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--teacher_ckpt_path", type=str, required=True)
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--time_scale", type=int, default=1)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=100000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    return parser


def train_model(args, make_student_model, make_teacher_model, make_dataset, device):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    train_dataset = InfinityDataset(make_dataset(), args.num_iters)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    student_ema = make_student_model().to(device)

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    def make_scheduler():
        M = importlib.import_module("train_utils")
        D = getattr(M, args.scheduler)
        return D()

    scheduler = make_scheduler()

    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        return D(model, betas, time_scale=time_scale)

    student = make_student_model().to(device)
    student_ema = make_student_model().to(device)
    teacher = make_teacher_model().to(device)

    teacher_ckpt = torch.load(args.teacher_ckpt_path)
    teacher.load_state_dict(teacher_ckpt["G"])
    del teacher_ckpt
    print("Teacher loaded...")

    for p in teacher.parameters(): # Freeze teacher
        p.required_grad = False

    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        student.load_state_dict(ckpt["G"])
        student_ema.load_state_dict(ckpt["G"])
        del ckpt
        print("Continue training...")
    else:
        print("Training new model...")
    init_ema_model(student, student_ema)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    teacher_diffusion = make_diffusion(teacher, args.num_timesteps, args.time_scale, device)
    student_diffusion = make_diffusion(student, args.num_timesteps, args.time_scale, device)
    student_ema_diffusion = make_diffusion(student, args.num_timesteps, args.time_scale, device)

    image_size = student.image_size

    on_iter = make_iter_callback(student_ema_diffusion, device, checkpoints_dir, image_size, 
                                 tensorboard, args.log_interval, args.ckpt_interval, False)
    diffusion_train = QDiffusionTrain(scheduler, student_diffusion, teacher_diffusion, student_ema)
    diffusion_train.train(train_loader, args.lr, device, on_iter)
    print("Finished.")
    return student_diffusion.net_

class QDiffusionTrain:

    def __init__(self, scheduler, student_diffusion, teacher_diffusion, student_ema):
        self.scheduler = scheduler
        self.student_diffusion = student_diffusion
        self.teacher_diffusion = teacher_diffusion
        self.student_ema = student_ema
    
    def student_loss(self, x_0, t): 
        noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.teacher_diffusion.get_alpha_sigma(x_0, t)
        z_t = alpha_t * x_0 + sigma_t * noise
        s_zt = self.student_diffusion.inference(z_t.float(), t.float(), {})
        t_zt = self.teacher_diffusion.inference(z_t.float(), t.float(), {})
        return torch.nn.functional.mse_loss(s_zt, t_zt)

    def train(self, train_loader, model_lr, device, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(train_loader)
        scheduler.init(self.student_diffusion, model_lr, total_steps)
        self.student_diffusion.net_.train()
        print(f"Training...")
        pbar = tqdm(train_loader)
        N = 0
        L_tot = 0
        for img, _ in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = torch.randint(0, self.student_diffusion.num_timesteps, 
                                 (img.shape[0],), device=device)
            loss = self.student_loss(img, time)
            L_tot += loss.item()
            N += 1
            pbar.set_description(f"Loss: {L_tot / N:.5f}")
            loss.backward()
            nn.utils.clip_grad_norm_(self.student_diffusion.net_.parameters(), 1)
            scheduler.step()
            moving_average(self.student_diffusion.net_, self.student_ema)
            on_iter(N, loss.item())
            if scheduler.stop(N, total_steps):
                break
        on_iter(N, loss.item(), last=True)