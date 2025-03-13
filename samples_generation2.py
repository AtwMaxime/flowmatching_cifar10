import os
import sys
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from absl import app, flags

# If not already imported:
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

# UNet Parameters
flags.DEFINE_integer("num_channel", 128, help="Base channel of UNet")

# Evaluation Parameters
flags.DEFINE_string("input_dir", "./results", help="Output directory")
flags.DEFINE_string("integration_method", "euler", help="ODE solver method")
flags.DEFINE_integer("img_size", 32, help="Size of images")
flags.DEFINE_integer("num_interpolation", 20, help="Number of interpolation steps")
flags.DEFINE_string("model_name", "otfm", help="Name of model directory")
flags.DEFINE_integer("step", 400000, help="Which step to load from checkpoint")
flags.DEFINE_string("output_dir", "./generated_images", help="Where to save figures")

FLAGS(sys.argv)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name, step):
    """Load a trained UNet from checkpoint at a specific step."""
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

def generate_interpolation_images(model, num_steps=8):
    """
    1) Sample two random noise tensors x0, x1 (each shape: [1, 3, img_size, img_size]).
    2) Interpolate between x0 and x1 for num_steps values of t in [0, 1].
    3) Run the model (ODE solver) on each interpolated noise to get final images.
    Returns a list of final images for each interpolation step.
    """
    # Create two distinct noise seeds (batch=1 for simplicity).
    x0 = torch.randn(1, 3, FLAGS.img_size, FLAGS.img_size, device=device)
    x1 = torch.randn(1, 3, FLAGS.img_size, FLAGS.img_size, device=device)

    # Linspace for interpolation
    t_values = torch.linspace(0, 1, num_steps, device=device)

    final_images = []
    model.eval()
    with torch.no_grad():
        for t in t_values:
            # Interpolate in noise space
            x_t = (1 - t) * x0 + t * x1  # shape [1, 3, img_size, img_size]

            # Solve the ODE from time=0 to time=1 on x_t
            t_span = torch.linspace(0, 1, 10, device=device)  # e.g., 10 steps
            traj = odeint(model, x_t, t_span, method=FLAGS.integration_method)
            x_final = traj[-1]  # final state: shape [1, 3, H, W]

            # Convert to uint8 for easy viewing
            x_final = (x_final * 127.5 + 128).clip(0, 255).to(torch.uint8)
            final_images.append(x_final.squeeze(0))  # remove batch dim
    return final_images

def save_interpolation_plot(images, output_path):
    """
    Plot the images in a horizontal row, showing the progression of interpolation.
    """
    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=(2*n, 2))
    for i, img in enumerate(images):
        # shape: [3, H, W], we want to transpose -> [H, W, 3]
        ax[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[i].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def main(_):
    # 1) Load the model from checkpoint
    model = load_model(FLAGS.model_name, FLAGS.step)
    if model is None:
        return

    # 2) Generate interpolation images
    interp_images = generate_interpolation_images(model, FLAGS.num_interpolation)

    # 3) Save the interpolation result (all images in one row)
    output_path = os.path.join(FLAGS.output_dir,
                               f"interp_step_{FLAGS.step}.png")
    save_interpolation_plot(interp_images, output_path)
    print(f"✅ Saved interpolation figure to {output_path}")

if __name__ == "__main__":
    app.run(main)
