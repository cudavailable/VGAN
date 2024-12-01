import os
import argparse
from inference import infer

from train import train
from dataset import test

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_classes", type=int, default=5)
	parser.add_argument("--epochs", type=int, default=50)
	# parser.add_argument("--e_lr", type=float, default=0.001)
	parser.add_argument("--g_lr", type=float, default=0.0002)
	parser.add_argument("--d_lr", type=float, default=0.0002)
	parser.add_argument("--img_size", type=int, default=128)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--input_channel", type=int, default=3)
	parser.add_argument("--latent_size", type=int, default=200)
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--log_dir", type=str, default="./log")
	parser.add_argument("--log_path", type=str, default="log5.txt")
	parser.add_argument("--model_path", type=str, default="cvae_model.pth")
	parser.add_argument("--D_path", type=str, default="discriminator.pth")
	# parser.add_argument("--model_path", type=str, default="./data/saved_models/saved_model.tar")
	parser.add_argument("--recon_dir", type=str, default="./recon")
	parser.add_argument("--w_recon", type=float, default=4.5)
	parser.add_argument("--w_kl", type=float, default=1.0)
	# parser.add_argument("--w_loss_g", type=float, default=0.01)
	# parser.add_argument("--w_loss_gd", type=float, default=1.0)
	parser.add_argument("--w_adv", type=float, default=15.0)
	parser.add_argument("--gd_ratio", type=int, default=5)
	# (0.4914, 0.4822, 0.4465)
	# (0.247, 0.243, 0.261)
	parser.add_argument("--preTrain", type=bool, default=True)
	parser.add_argument("--tran_mean", type=tuple, default=(0.485, 0.456, 0.406))
	parser.add_argument("--tran_std", type=tuple, default=(0.229, 0.224, 0.225))

	args = parser.parse_args()

	# train(args)
	infer(args)