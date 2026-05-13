# VQ-VAE with Autoregressive Prior on MNIST

This project was developed as part of the course **Generative Deep Learning (708.088)** at Graz University of Technology.  
The goal is to train a **Vector Quantized Variational Autoencoder (VQ-VAE)** on MNIST and combine it with an **autoregressive prior** over the discrete latent representation.

## Project Overview

The model learns a discrete latent representation of handwritten digit images. First, a VQ-VAE is trained to reconstruct MNIST images using a learned codebook. Afterwards, an autoregressive model is trained on the discrete latent codes produced by the VQ-VAE. The full model can then generate new digit-like samples by sampling latent code sequences from the autoregressive prior and decoding them with the VQ-VAE decoder.

## Main Components

- **Encoder:** maps flattened MNIST images to latent vectors.
- **Vector Quantizer:** replaces continuous latent vectors with nearest entries from a learned discrete codebook.
- **Decoder:** reconstructs the original image from the quantized latent representation.
- **Autoregressive Prior:** models the distribution of discrete latent code sequences using causal 1D convolutions.
- **Sampling Pipeline:** generates new latent code sequences and decodes them into images.

## Methods Used

- Vector Quantized Variational Autoencoder (VQ-VAE)
- Discrete latent representations
- Codebook learning and commitment loss
- Straight-through gradient estimator
- Autoregressive modeling with causal convolutions
- Training/validation loss evaluation
- Image reconstruction and sample visualization

## Dataset

The project uses the **MNIST** handwritten digit dataset. The dataset is downloaded automatically through `torchvision.datasets.MNIST` when running the notebook.

## Technologies

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook Assignment1_GDL-SS26.ipynb
```

4. Run the notebook cells in order.

The MNIST dataset will be downloaded automatically into the `data/` directory.

## Outputs

The notebook produces:

- VQ-VAE validation loss curves
- Original vs. reconstructed MNIST images
- Autoregressive prior validation loss curves
- Generated samples from the trained autoregressive prior and VQ-VAE decoder
- Saved model files in the `results/` directory

## Notes

This repository is intended as an academic machine learning project. The focus is on understanding and implementing the main components of a VQ-VAE and an autoregressive prior rather than building a production-level generative model.
