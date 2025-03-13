import os
import copy
import torch
from torchvision import datasets, transforms
from tqdm import trange
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import ema, generate_samples, infiniteloop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_flow_matching(model, optimizer, datalooper, fm_method, total_steps, save_step, savedir):
    """ Train the flow matching model with or without OT sampling. """
    ema_model = copy.deepcopy(model)
    FM = fm_method(sigma=0.0)

    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema(model, ema_model, 0.9999)

            if save_step > 0 and step % save_step == 0:
                generate_samples(model, False, savedir, step, net_="normal")
                generate_samples(ema_model, False, savedir, step, net_="ema")
                torch.save({
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }, os.path.join(savedir, f"cifar10_weights_step_{step}.pt"))


if __name__ == "__main__":
    batch_size = 128
    total_steps = 200000
    save_step = 20000

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)
    datalooper = infiniteloop(dataloader)

    model_fm = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=128, channel_mult=[1, 2, 2, 2],
        num_heads=4, num_head_channels=64, attention_resolutions="16", dropout=0.1
    ).to(device)
    optimizer_fm = torch.optim.Adam(model_fm.parameters(), lr=2e-4)
    os.makedirs("./results/fm", exist_ok=True)
    train_flow_matching(model_fm, optimizer_fm, datalooper, ConditionalFlowMatcher, total_steps, save_step,
                        "./results/fm")

    model_otfm = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=128, channel_mult=[1, 2, 2, 2],
        num_heads=4, num_head_channels=64, attention_resolutions="16", dropout=0.1
    ).to(device)
    optimizer_otfm = torch.optim.Adam(model_otfm.parameters(), lr=2e-4)
    os.makedirs("./results/otfm", exist_ok=True)
    train_flow_matching(model_otfm, optimizer_otfm, datalooper, ExactOptimalTransportConditionalFlowMatcher,
                        total_steps, save_step, "./results/otfm")