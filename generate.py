import os
import torch
import torchvision
import sys
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchdiffeq import odeint
from tqdm import tqdm
from PIL import Image
from absl import app, flags
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

# UNet Parameters
flags.DEFINE_integer("num_channel", 128, help="Base channel of UNet")

# Evaluation Parameters
flags.DEFINE_string("input_dir", "./results", help="Output directory")
flags.DEFINE_list("nfes", ["1", "2", "5", "10", "50"], help="List of NFEs to evaluate")
flags.DEFINE_string("integration_method", "euler", help="ODE solver method")
flags.DEFINE_integer("step", 180000, help="Checkpoint step to evaluate")
flags.DEFINE_integer("num_gen", 30000, help="Number of images to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance")
flags.DEFINE_integer("batch_size_fid", 2048, help="Batch size for FID computation")

FLAGS(sys.argv)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    """ Load a trained model from checkpoint """
    print(f"🔄 Loading model: {model_name}")
    
    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    path = f"{FLAGS.input_dir}/{model_name}/cifar10_weights_step_{FLAGS.step}.pt"
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return None

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["ema_model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_and_save_images(model, nfe, output_dir, num_images=30000):
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_images, FLAGS.batch_size_fid), desc=f"Generating images for NFE {nfe}"):
            batch_size = min(FLAGS.batch_size_fid, num_images - i)
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            t_span = torch.linspace(0, 1, nfe + 1, device=device)
            traj = odeint(model, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method)
            imgs = (traj[-1, :] * 127.5 + 128).clip(0, 255).to(torch.uint8)

            for j, img in enumerate(imgs):
                torchvision.utils.save_image(img.float() / 255.0, f"{output_dir}/{i+j:05d}.png")

def load_generated_images(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        img = Image.open(os.path.join(directory, filename)).convert("RGB")
        img = transform(img)
        images.append(img)
    return torch.stack(images).to(torch.uint8)

def compute_fid(generated_images, cifar10_loader):
    fid = FrechetInceptionDistance(feature=2048)
    
    for real_batch, _ in cifar10_loader:
        fid.update(real_batch, real=True)
    
    for i in range(0, len(generated_images), FLAGS.batch_size_fid):
        batch = generated_images[i:i + FLAGS.batch_size_fid]
        fid.update(batch, real=False)
    
    return fid.compute().item()

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8
])

cifar10_data = CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar10_loader = DataLoader(cifar10_data, batch_size=FLAGS.batch_size_fid, shuffle=False, num_workers=4)

# # Load the model
model_name = "fm"
model = load_model(model_name)
if model is None:
    raise RuntimeError("Model could not be loaded.") 

# Save FID results
fid_results_path = "./log/fid_results.txt"
with open(fid_results_path, "w") as fid_file:
    for nfe in map(int, FLAGS.nfes):
        output_dir = f"./log/NFE_{nfe}"
        generate_and_save_images(model, nfe, output_dir, num_images=30000)
        generated_images = load_generated_images(output_dir)
        fid_score = compute_fid(generated_images, cifar10_loader)
        print(f"FID Score for NFE {nfe}: {fid_score}")
        fid_file.write(f"NFE {nfe}: {fid_score}\n")