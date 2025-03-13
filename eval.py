import os
import sys
import torch
from absl import app, flags
from torchdiffeq import odeint
from torchcfm.models.unet.unet import UNetModelWrapper
from test import compute_fid

FLAGS = flags.FLAGS

# UNet
flags.DEFINE_integer("num_channel", 128, help="Base channel of UNet")

# Evaluation Parameters
flags.DEFINE_string("input_dir", "./results", help="Output directory")
flags.DEFINE_list("models", [ "otfm"], help="List of models to evaluate")
flags.DEFINE_list("nfes", ["1", "2", "5", "10", "20","50"], help="List of NFEs to evaluate")
flags.DEFINE_string("integration_method", "euler", help="ODE solver method")
flags.DEFINE_integer("step", 400000, help="Checkpoint step to evaluate")
flags.DEFINE_integer("num_gen", 35000, help="Number of images to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 2048, help="Batch size for FID computation")

FLAGS(sys.argv)

# Define the device
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

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    return model


def gen_images(model, nfe):
    """ Generate images using the trained model for a given NFE """
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        t_span = torch.linspace(0, 1, nfe + 1, device=device)  # Adjust time steps based on NFE
        traj = odeint(model, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method)

    traj = traj[-1, :]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img


# Compute FID for each model at different NFEs
fid_scores = {}

for model_name in FLAGS.models:
    model = load_model(model_name)
    if model is None:
        continue

    fid_scores[model_name] = {}

    for nfe in FLAGS.nfes:
        nfe = int(nfe)
        print(f"📊 Computing FID for {model_name} with NFE={nfe}...")

        score = compute_fid(
            gen=lambda _: gen_images(model, nfe),
            dataset_name="cifar10",
            dataset_res=32,
            num_gen=FLAGS.num_gen,
            batch_size=FLAGS.batch_size_fid,
            dataset_split="train"
        )

        fid_scores[model_name][nfe] = score
        print(f"✅ FID for {model_name} at NFE={nfe}: {score}")

# Save results to a file
results_path = os.path.join(FLAGS.input_dir, "fid_results.txt")

with open(results_path, "w") as f:
    f.write("=== FID Scores for Different NFEs ===\n")
    for model_name, nfe_scores in fid_scores.items():
        f.write(f"\nModel: {model_name}\n")
        for nfe, score in nfe_scores.items():
            f.write(f"  NFE {nfe}: {score}\n")

print(f"\n📄 FID scores saved to {results_path}")
