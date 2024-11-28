import os
import torch
from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CVAE
from logger import Logger


def train(args):

	# logger setup
	if args.log_dir is not None and not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	logger = Logger(os.path.join(args.log_dir, "log.txt"))

	# keep the best model parameters according to avg_loss
	tracker = {"epoch" : None, "criterion" : None}

	# device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.write(f"we're using :: {device}\n\n")

	# data preparations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		# transforms.Lambda(lambda x : x.view(-1)) # flatten the 28x28 image to 1D
	])
	cifar = CIFAR10(args.data_dir, train=True, transform=transform, download=True)
	dataset = DataLoader(dataset=cifar, batch_size=args.batch_size, shuffle=True)

	# loss function for cvae
	def loss_fn(x_recon, x, mean, log):
		recon_loss = nn.MSELoss()(x_recon, x)
		kl_div = -0.5 * torch.sum(1 + log - mean.pow(2) - log.exp())
		return (recon_loss + kl_div) / x.size(0)

	# model setup
	# need to be fixed...
	model = CVAE(input_channel=args.input_channel, condition_dim=args.num_classes, latent_dim=args.latent_size).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Training
	model.train()
	for epoch in range(args.epochs):

		epoch_loss = 0
		for x, y in dataset:
			x, y = x.to(device), y.to(device)
			c = nn.functional.one_hot(y, num_classes=args.num_classes).float().to(device) # one-hot encoding

			# update gradients
			optimizer.zero_grad()
			x_recon, m, log = model(x, c)
			loss = loss_fn(x_recon, x, m, log)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		# keep the best model parameters according to avg_loss
		avg_loss = epoch_loss / len(dataset)
		# tracker = {"epoch" : None, "criterion" : None, "model_params" : None}
		if tracker["criterion"] is None or avg_loss < tracker["criterion"]:
			tracker["epoch"] = epoch + 1
			tracker["criterion"] = avg_loss
			torch.save(model.state_dict(), args.model_path)
			pass
		logger.write(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}\n")

	# end Training
	logger.write("\n\nTraining completed\n\n")
	logger.write(f"Best Epoch: {tracker['epoch']}, average loss:{tracker['criterion']}\n")
	if tracker["model_params"] is not None:
		logger.write(f"model with the best performance saved to {args.model_path}.")
	# close
	logger.close()