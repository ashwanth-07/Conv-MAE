# Conv-MAE

Self-supervised **masked-autoencoder (MAE) pre-training** for convolutional / hybrid vision backbones, using an [**EfficientViT**](https://github.com/mit-han-lab/efficientvit) encoder and a lightweight multi-scale Transformer decoder.

The model masks a large fraction of the input, encodes the visible content with a multi-stage EfficientViT backbone, fuses the backbone's multi-scale feature maps, and reconstructs the missing image patches. After pre-training, the encoder can be transferred to downstream tasks (classification, detection, segmentation).

---

## How it works

```
        input image (B, 3, H, W)
                 │
   generate_mask │  (random patch mask at H/32 resolution, mask_ratio=0.75)
                 ▼
        EfficientViT backbone (encoder)
                 │   produces 4 multi-scale feature maps
                 │   stage1 (H/4) · stage2 (H/8) · stage3 (H/16) · stage4 (H/32)
                 ▼
        ConvMAEDecoder
          1. project each stage to a common decoder_dim and a common
             spatial size (H/16) via strided / 1x1 convs + interpolation
          2. fuse the 4 projected scales with an MLP
          3. replace masked positions with a learned mask token
          4. add 2D sin-cos positional encoding
          5. run Transformer blocks
          6. linear head -> per-patch pixel predictions
                 ▼
        reconstruction loss (MSE on masked patches only)
```

Key components live in [`convmae_models.py`](convmae_models.py):

- `TransformerBlock` — pre-norm multi-head self-attention + MLP decoder block.
- `PositionalEncoding2D` — fixed 2D sin-cos positional embedding.
- `ConvMAEDecoder` — multi-scale projection, fusion, masked-token decoding, and `forward_loss` (optionally with per-patch normalized pixel targets, `norm_pix_loss`).
- `ConvMAEPretrainer` — wraps an EfficientViT backbone + decoder into a single module that returns `(pred, loss, mask)`.

The masking itself is implemented in the backbone (`generate_mask` in [`efficientvit/models/efficientvit/backbone.py`](efficientvit/models/efficientvit/backbone.py)) so the mask is applied consistently across the convolutional stages.

---

## Repository structure

```
.
├── convmae_models.py      # ConvMAE decoder + pretrainer model definitions
├── pretrain.py            # Pre-training entry point (data, train/val loop, logging, checkpoints)
├── profiler.py            # Per-component timing / throughput / memory profiler
├── efficientvit/          # Vendored EfficientViT backbone, layers, and model utilities
│   ├── models/
│   │   └── efficientvit/
│   │       └── backbone.py
│   └── ...
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.9+ and a CUDA-capable GPU for realistic training (CPU works for the profiler and small smoke tests).

```bash
git clone https://github.com/ashwanth-07/Conv-MAE.git
cd Conv-MAE

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Install the [PyTorch build](https://pytorch.org/get-started/locally/) that matches your CUDA version if the default wheels don't suit your setup.

---

## Dataset layout

`pretrain.py` reads data with `torchvision.datasets.ImageFolder`, so point `--data_path` at a directory containing `train/` and `val/` splits, each with one subfolder per class:

```
<data_path>/
├── train/
│   ├── class_a/
│   │   └── *.jpg
│   └── class_b/
│       └── *.jpg
└── val/
    ├── class_a/
    └── class_b/
```

Labels are ignored (pre-training is self-supervised); only the folder structure matters. Images are resized/cropped to `--image_size` (default 224) and normalized with ImageNet statistics.

---

## Usage

### Pre-training

```bash
python pretrain.py \
    --backbone b2 \
    --data_path /path/to/imagenet \
    --batch_size 256 \
    --epochs 16 \
    --mask_ratio 0.75 \
    --output_dir ./output_convmae
```

Common arguments (see `python pretrain.py --help` for the full list):

| Argument | Default | Description |
| --- | --- | --- |
| `--backbone` | `b2` | EfficientViT backbone: `b0`, `b1`, `b2`, `b3` |
| `--mask_ratio` | `0.75` | Fraction of patches masked |
| `--decoder_dim` | `512` | Decoder embedding dimension |
| `--decoder_depth` | `8` | Number of decoder Transformer blocks |
| `--patch_size` | `16` | Reconstruction patch size |
| `--norm_pix_loss` | off | Use per-patch normalized pixel targets |
| `--batch_size` | `256` | Batch size per GPU |
| `--epochs` | `16` | Number of pre-training epochs |
| `--warmup_epochs` | `4` | Linear LR warmup epochs |
| `--lr` | `1.5e-4` | Peak learning rate (cosine schedule) |
| `--weight_decay` | `0.05` | AdamW weight decay |
| `--image_size` | `224` | Input resolution |
| `--num_workers` | `8` | Dataloader workers |
| `--output_dir` | `./output_convmae` | Where logs, checkpoints, and visualizations are written |
| `--resume` | `''` | Path to a checkpoint to resume from |
| `--device` | `cuda` | `cuda` or `cpu` |

### Outputs

Inside `--output_dir` you will find:

- `args.json` — the exact run configuration.
- `convmae_training_train.csv`, `convmae_training_val.csv`, `convmae_training_general.txt` — structured training logs.
- `training_curves.png` — loss and learning-rate curves.
- `visualizations/` — side-by-side original / masked / reconstructed image grids.
- `checkpoint_epoch_*.pth`, `checkpoint_best.pth`, `checkpoint_final.pth` — model + optimizer state.

### Profiling

`profiler.py` measures per-component latency, throughput, and (on CUDA) memory using synthetic data — no dataset required:

```bash
python profiler.py --backbone b2 --batch_size 8 --iterations 20 --device cuda
```

It prints a timing breakdown, writes a JSON report, and saves a timing plot to `--output_dir` (default `./timing_results`). Use `--device cpu`, `--no_memory`, or `--no_plot` as needed.

---

## Model variants

| Backbone | Encoder stage widths | Notes |
| --- | --- | --- |
| `b0` | 16 / 32 / 64 / 128 | Smallest, fastest |
| `b1` | 32 / 64 / 128 / 256 | |
| `b2` | 48 / 96 / 192 / 384 | Default |
| `b3` | 64 / 128 / 256 / 512 | Largest of the supported set |

The vendored `efficientvit` package also defines larger `l0`–`l3` backbones, but the `ConvMAEPretrainer` currently wires up the `b0`–`b3` variants.

---

## Acknowledgements

This project builds on the [EfficientViT](https://github.com/mit-han-lab/efficientvit) backbone and is inspired by [Masked Autoencoders Are Scalable Vision Learners (He et al.)](https://arxiv.org/abs/2111.06377) and [ConvMAE](https://arxiv.org/abs/2205.03892). Portions of the vendored backbone carry an AMD modification copyright (see file headers). Please consult the upstream EfficientViT repository for its original license terms before redistribution.
