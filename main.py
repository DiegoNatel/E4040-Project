"""Entry point that wires datasets, models, and the trainer to run DCGAN experiments."""

import argparse
from pathlib import Path

from data_utils import get_dataset
from model import (
    Discriminator,
    Discriminator32,
    Generator,
    Generator32,
)
from train_dcgan import DCGANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN models end to end.")
    parser.add_argument(
        "--dataset",
        choices=["faces", "lsun", "imagenet"],
        default="faces",
        help="Dataset to train on (CelebA faces, LSUN bedrooms, ImageNet 32×32).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local directory that stores CelebA images (required for faces).",
    )
    parser.add_argument(
        "--lsun-category",
        type=str,
        default="bedroom",
        help="LSUN category when --dataset=lsun.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to run."
    )
    parser.add_argument(
        "--z-dim", type=int, default=100, help="Dimensionality of the noise vector."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="generated_faces",
        help="Directory where sample grids are stored.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for TensorFlow checkpoints and weights.",
    )
    parser.add_argument(
        "--g-lr", type=float, default=2e-4, help="Generator learning rate."
    )
    parser.add_argument(
        "--d-lr", type=float, default=2e-4, help="Discriminator learning rate."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore from the latest checkpoint before training.",
    )
    return parser.parse_args()


def build_dataset(args):
    base_kwargs = {"batch_size": 128, "shuffle": True}

    dataset = args.dataset
    if dataset == "faces":
        if not args.data_dir:
            raise ValueError("--data-dir is required when training on CelebA faces.")
        base_kwargs["data_dir"] = args.data_dir
        base_kwargs["target_size"] = (64, 64)
    elif dataset == "lsun":
        base_kwargs["category"] = args.lsun_category
        base_kwargs["target_size"] = (64, 64)
        base_kwargs["deduplicate"] = True
    elif dataset == "imagenet":
        base_kwargs["target_size"] = (32, 32)
        base_kwargs["subset"] = "train"
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    return get_dataset(dataset_name=dataset, **base_kwargs)


def main():
    args = parse_args()
    dataset = build_dataset(args)

    if args.dataset == "imagenet":
        generator = Generator32(z_dim=args.z_dim)
        discriminator = Discriminator32()
    else:
        generator = Generator(z_dim=args.z_dim)
        discriminator = Discriminator()

    trainer = DCGANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        z_dim=args.z_dim,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
    )

    trainer.train(epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()

"""
main.py

Reproducing paper: 
UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS


- Load and preprocess image datasets.
- Construct the DCGAN generator and discriminator
- Train the model to generate 64×64 images
- Optionally extract discriminator convolutional features for downstream
  tasks such as CIFAR-10 or SVHN classification (as done in the paper).
- Checkpoints, logs, and reproducibility.

Run this script to train a DCGAN model.
"""




