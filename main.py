import os
import argparse
from inference import infer

from train import train

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_classes", type=int, default=10)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--input_channel", type=int, default=3)
	parser.add_argument("--latent_size", type=int, default=128)
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--log_dir", type=str, default="./log")
	parser.add_argument("--model_path", type=str, default="cvae_model.pth")
	parser.add_argument("--recon_dir", type=str, default="./recon")

	args = parser.parse_args()

	# train(args)
	infer(args)