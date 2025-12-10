# ------------------------------------------------------------------------------
# File: train_dcgan.py
# This implementation is partially based on the official TensorFlow DCGAN tutorial:
# https://www.tensorflow.org/tutorials/generative/dcgan
# ------------------------------------------------------------------------------

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DCGANTrainer:
    """Encapsulates full DCGAN training (losses, checkpoints, and samples)."""

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        dataset: tf.data.Dataset,
        z_dim: int = 100,
        checkpoint_dir: str = "checkpoints",
        sample_dir: str = "generated_faces",
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
        beta_1: float = 0.5,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.z_dim = z_dim
        self.sample_dir = Path(sample_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=beta_1)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=3,
        )

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        z = tf.random.uniform([batch_size, self.z_dim], minval=-1.0, maxval=1.0)

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_images = self.generator(z, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            d_loss = self._discriminator_loss(real_output, fake_output)
            g_loss = self._generator_loss(fake_output)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

    def generate_and_save_images(self, epoch, grid_size: int = 4):
        num_images = grid_size * grid_size
        z = tf.random.uniform([num_images, self.z_dim], -1.0, 1.0)
        predictions = self.generator(z, training=False)
        imgs = ((predictions + 1.0) * 127.5).numpy().astype("uint8")

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(imgs[idx])
            ax.axis("off")
        plt.tight_layout()

        sample_path = self.sample_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(sample_path)
        plt.close(fig)

    def train(self, epochs: int, resume: bool = False):
        if resume and self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored checkpoint: {self.checkpoint_manager.latest_checkpoint}")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            d_losses = []
            g_losses = []

            for real_images in self.dataset:
                d_loss, g_loss = self.train_step(real_images)
                d_losses.append(d_loss.numpy())
                g_losses.append(g_loss.numpy())

            print(
                f"  d_loss: {np.mean(d_losses):.4f}, g_loss: {np.mean(g_losses):.4f}"
            )
            self.generate_and_save_images(epoch)
            self.checkpoint_manager.save()
            self.generator.save_weights(
                str(self.checkpoint_dir / "generator_weights.h5")
            )
            self.discriminator.save_weights(
                str(self.checkpoint_dir / "discriminator_weights.h5")
            )
