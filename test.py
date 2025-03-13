import os
import numpy as np
import torch
from cleanfid.fid import get_model_features
from cleanfid.fid import frechet_distance

from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_model_features

def get_reference_statistics(dataset_name, dataset_res, dataset_split="train"):
    """
    Load precomputed reference statistics (mu, sigma) from a hardcoded list of paths.
    """
    possible_files = [
        f"/local_scratch/mattwood/cfm_cifar10/data/cleanfid_stats/{dataset_name}_legacy_tensorflow_{dataset_split}_{dataset_res}.npz",
        f"/scratch/chamaeleon/mattwood/cfm_cifar10/data/cleanfid_stats/{dataset_name}_legacy_tensorflow_{dataset_split}_{dataset_res}.npz",
        f"/data/cleanfid_stats/{dataset_name}_legacy_tensorflow_{dataset_split}_{dataset_res}.npz",
    ]

    stats_file = next((path for path in possible_files if os.path.exists(path)), None)

    if stats_file is None:
        raise FileNotFoundError(f"🚨 ERROR: No valid CleanFID statistics file found!\nChecked paths:\n" + "\n".join(possible_files))

    print(f"📂 Using CleanFID stats from: {stats_file}")

    stats = np.load(stats_file)
    return stats["mu"], stats["sigma"]




def fid_model(G, dataset_name, dataset_res, dataset_split,
              model=None, mode="legacy_tensorflow", num_gen=50_000, batch_size=128,
              device=torch.device("cuda"), verbose=True):
    """
    Compute FID score for a model using precomputed dataset statistics.
    """
    # Load reference FID statistics
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res, dataset_split)

    # ✅ Force feature extractor to be "legacy_tensorflow" (to match dataset stats)
    feat_model = build_feature_extractor(mode="legacy_tensorflow", device=device)

    # ✅ Get model features using the correct feature extractor
    model_features = get_model_features(G, feat_model, mode="clean", num_gen=num_gen, batch_size=batch_size, device=device)

    print("✅ Extracted Model Features Shape:", model_features.shape)  # Debugging output

    # ✅ Ensure features are 2D (samples, feature_dim)
    if len(model_features.shape) > 2:
        print("🚨 Warning: Model features are not 2D. Reshaping...")
        model_features = model_features.reshape(model_features.shape[0], -1)

    # Compute FID
    mu = np.mean(model_features, axis=0)
    sigma = np.cov(model_features, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid



def compute_fid(fdir1=None, fdir2=None, gen=None,
                dataset_name="cifar10", dataset_res=32, dataset_split="train",
                num_gen=50_000, batch_size=128, device=torch.device("cuda")):
    """
    Computes the FID for a model using reference statistics.
    """
    if gen is None:
        raise ValueError("🚨 ERROR: No model (gen) provided for FID computation!")

    # ✅ Pass the generator model for evaluation
    return fid_model(gen, dataset_name, dataset_res, dataset_split, model=gen, num_gen=num_gen, batch_size=batch_size, device=device)
