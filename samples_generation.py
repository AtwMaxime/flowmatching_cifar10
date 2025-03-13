import os
import torch
import torchvision
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
from torchdiffeq import odeint
from tqdm import tqdm
from absl import app, flags
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

# UNet Parameters
flags.DEFINE_integer("num_channel", 128, help="Base channel of UNet")

# Evaluation Parameters
flags.DEFINE_string("input_dir", "./results", help="Output directory")
flags.DEFINE_list("steps", ["20000","60000", "120000", "180000"], help="Steps to evaluate")
flags.DEFINE_string("integration_method", "euler", help="ODE solver method")
flags.DEFINE_integer("num_images", 10, help="Number of images to generate")
flags.DEFINE_integer("img_size", 32, help="Size of images")

FLAGS(sys.argv)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name, step):
    """ Load a trained model from checkpoint at a specific step """
    print(f"🔄 Loading model: {model_name} at step {step}")
    
    model = UNetModelWrapper(
        dim=(3, FLAGS.img_size, FLAGS.img_size),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    path = f"{FLAGS.input_dir}/{model_name}/cifar10_weights_step_{step}.pt"
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return None

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["ema_model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_images(model, num_images=10):
    """ Generate a set number of images """
    with torch.no_grad():
        x = torch.randn(num_images, 3, FLAGS.img_size, FLAGS.img_size, device=device)
        t_span = torch.linspace(0, 1, 10, device=device)
        traj = odeint(model, x, t_span, method=FLAGS.integration_method)
        imgs = (traj[-1, :] * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return imgs

def save_image_plot(images, step, output_dir):
    """ Save a row of generated images """
    fig, ax = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        ax[i].imshow(img)
        ax[i].axis("off")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/generated_step_{step}.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    model_name = "otfm"
    output_dir = "./generated_images"

    for step in FLAGS.steps:
        model = load_model(model_name, step)
        if model:
            images = generate_images(model, FLAGS.num_images)
            save_image_plot(images, step, output_dir)