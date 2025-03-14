# Flow Matching on CIFAR-10

This repository contains code for training and evaluating Flow Matching (FM) and Optimal Transport Flow Matching (OT-FM) models on the CIFAR-10 dataset.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Generating Samples](#generating-samples)
- [Evaluating Models](#evaluating-models)
- [Scripts Overview](#scripts-overview)

## Dataset

The CIFAR-10 dataset is automatically downloaded by the training script. It consists of 60,000 images (50,000 training, 10,000 testing) of size 32×32 pixels across 10 classes.

## Training

To train both FM and OT-FM models:

```bash
python main.py
```

- The script trains two models:
  - **Flow Matching (FM)**
  - **Optimal Transport Flow Matching (OT-FM)**
- Models are saved in `./results/fm/` and `./results/otfm/` at regular intervals.

## Generating Samples

To generate and save images from trained models:

```bash
python samples_generation.py
```

To generate interpolated images:

```bash
python samples_generation2.py
```

## Evaluating Models

To generate images and compute FID (Frechet Inception Distance):

```bash
python generate.py
```

To evaluate trained models at different NFEs:

```bash
python eval.py
```

---

## Scripts Overview

### **1. `utils_cifar10.py`**
Utility functions for training and sampling:
- `setup()`: Initializes the distributed training environment.
- `generate_samples()`: Saves 64 generated images during training for sanity check.
- `ema()`: Implements Exponential Moving Average (EMA) for model parameters.
- `infiniteloop()`: Creates an infinite loop iterator over the dataloader.

---

### **2. `main.py`**
Trains both FM and OT-FM models:
- Loads the CIFAR-10 dataset.
- Defines and initializes the U-Net model.
- Uses Conditional Flow Matching and Optimal Transport FM for training.
- Saves models at intervals.

---

### **3. `samples_generation.py`**
Generates and saves images from trained models:
- Loads models from checkpoints.
- Uses an ODE solver to generate images from Gaussian noise.
- Saves results in `./generated_images/`.

---

### **4. `samples_generation2.py`**
Generates interpolation between two noise vectors:
- Loads a trained model.
- Interpolates between two random noise tensors.
- Runs the model at each interpolation step.
- Saves the results as a row of images.

---

### **5. `generate.py`**
Generates a large number of images and computes FID:
- Loads trained models.
- Generates images for multiple Numbers of Function Evaluations (NFEs).
- Computes FID by comparing generated images with CIFAR-10.

---

### **6. `eval.py`**
Evaluates trained models:
- Loads models at a specific training step.
- Generates images at different NFEs.
- Computes and logs FID scores.

---

## Results

- The generated images and their corresponding FID scores are stored in `./log/`.
- Intermediate training samples are saved in `./results/`.

---
