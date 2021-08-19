import os
import sys

from rlkit.core import logger
from rlkit.testing import csv_util

from examples.val.train_vqvae import main

def test_train_vqvae_fn():
    # running with GPU fails! must be some source of randomness in VQVAE with GPU
    cmd = "python examples/val/train_vqvae.py --1 --local --run 1 --debug --seed 0"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if VQVAE training results matches
    reference_csv = "tests/regression/val/id0_vqvae/vae_progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "vae_progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "test/Perplexity", "test/Recon Error", "test/VQ Loss", "test/losses", "train/Perplexity", "train/Recon Error", "train/VQ Loss", "train/losses"]
    csv_util.check_equal(reference, output, keys)

    # check if PixelCNN training results match
    reference_csv = "tests/regression/val/id0_vqvae/pixelcnn_progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pixelcnn_progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "num_train_batches", "test/loss", "train/loss"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_train_vqvae_fn()
