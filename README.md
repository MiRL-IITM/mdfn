# MDFN: Efficient Image Super-Resolution through Multi-Domain Feature Fusion

📄 **Paper Accepted: ICVGIP 2025**

This repository is the official implementation of our paper, "MDFN: Efficient Image Super-Resolution through Multi-Domain Feature Fusion".

## 💡 Abstract

Recent studies in image super-resolution have primarily employed transformer models, which require extensive training data and higher computational resources due to the quadratic complexity of self-attention. We propose MDFN (Multi-Domain Feature Network), a novel neural network for image super-resolution that efficiently combines spatial, multi-scale, and frequency feature representations. Unlike existing methods that rely heavily on transformer architectures or purely convolutional approaches, MDFN integrates three complementary domains: spatial feature extraction through convolutional layers, multi-scale features with Laplacian pyramid decomposition, and global frequency-aware features by Fourier domain processing. Our architecture leverages the inductive bias of convolutions while exploiting the global receptive field of frequency domain transformations to capture both local textures and global structural information. We evaluate MDFN on standard super-resolution benchmarks, including Set5, Set14, BSD100, and Urban100 datasets. Our experiments demonstrate that MDFN achieves competitive performance across all datasets, requiring up to 10× fewer parameters than the current state-of-the-art models.

## 🏗️ Architecture

MDFN is designed to create a comprehensive feature representation by concurrently processing information in three fundamental domains:

- **Spatial Domain:** Maps the input image from RGB space to a high-dimensional feature space using three sequential 3×3 convolutional layers with GELU activations. It leverages the strong inductive bias of convolutions to extract fundamental local patterns, edges, and textures efficiently.
- **Laplacian Pyramid Block (Multi-Scale Domain):** Explicitly models image structures across multiple resolutions through a Laplacian pyramid decomposition feature extractor. It computes high-frequency detail maps across multiple pyramid levels, refines them with wider receptive fields (7×7 convolutions), and unifies the multi-scale representations for robust signal fusion.
- **Fourier Block (Frequency Domain):** To model global context and long-range dependencies efficiently, it employs a Fast Fourier Convolution-based block. It splits features into a local branch (for fine-grained details via standard convolutions) and a global branch (utilizing Forward/Inverse FFTs, a learnable Position Embedding, and real-valued standard convolutions in the frequency domain) to explicitly ensure the generated features are coherent throughout the entire image.

The outputs from these three domains are concatenated and subsequently passed through a refinement stage employing a PixelShuffle layer and a global residual connection to reconstruct the final high-resolution outcome.

## 📊 Results

MDFN was evaluated against state-of-the-art architectures across several benchmark datasets without relying on large-scale pretraining.

**Quantitative Results (2× SR task):**

- **Set5:** 35.79 PSNR / 0.9533 SSIM
- **Set14:** 31.33 PSNR / 0.9074 SSIM
- **BSD100:** 32.93 PSNR / 0.9339 SSIM
- **Urban100:** 29.50 PSNR / 0.9085 SSIM

**Efficiency Advantanges:**
MDFN is remarkably lightweight and compact when compared to other SOTA models achieving similar performance for 2× SR task:

- **MDFN (Ours):** 2.91 M Parameters | 5.70 G FLOPs
- **SwinFIR:** 14.23 M Parameters | 51.17 G FLOPs
- **HAT-L:** 40.17 M Parameters | 165.53 G FLOPs
- **RCAN:** 15.44 M Parameters | 125.50 G FLOPs

## 🚀 How to Run Inference

You can generate super-resolved (HR) images from your own low-resolution (LR) inputs using the provided `inference.py` script.

**Example execution (for 2× scale):**

```bash
python inference.py -i data/test_image.png -s 2 -o data/output_hr.png
```

**Arguments:**

- `-i`, `--input`: Path to the input low-resolution image
- `-s`, `--scale`: Super-resolution scale factor (`2` or `4`)
- `-o`, `--output`: Path to save the output high-resolution image

The script will automatically detect if a CUDA device is available and load the corresponding checkpoint and YAML configuration (e.g. `config/MDFN_2X.yaml`).

## 🧪 How to Test

To evaluate the model's metrics on the benchmark datasets, execute the `test.py` script. The script calculates crucial performance indicators (PSNR, SSIM, FLOPS, Model Parameters) simulating evaluations across standard benchmarks.

**Run testing:**

```bash
python test.py
```

> **Note:** The scale factor for evaluation is directly configured in the script. You can modify the `SCALE_FACTOR = 2` (or `4`) variable inside `test.py` to test different scale factor checkpoints. The script fetches the standard valid data splits under `eugenesiow` benchmark datasets automatically.
