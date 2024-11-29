import os
import torch
from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CVAE, Discriminator
from logger import Logger

# loss function for cvae
def loss_G(x_recon, x, mean, log, fake_validity, real_labels, args):
	recon_loss = nn.MSELoss()(x_recon, x)
	kl_div = -0.5 * torch.sum(1 + log - mean.pow(2) - log.exp())
	adv_loss = nn.BCELoss()(fake_validity, real_labels)

	recon_loss *= args.w_recon
	kl_div *= args.w_kl
	adv_loss *= args.w_adv

	# weight to be fixed...
	print(f"recon: {recon_loss:.6f}, kl: {kl_div:.6f}, adv: {adv_loss:.6f}, x0: {x.size(0)}")
	return (recon_loss + kl_div + adv_loss) / x.size(0)

def loss_D(real_validity, real_labels, fake_validity, fake_labels):
	adversarial_loss = nn.BCELoss()
	d_real_loss = adversarial_loss(real_validity, real_labels)
	d_fake_loss = adversarial_loss(fake_validity, fake_labels)
	d_loss = (d_real_loss + d_fake_loss) * 0.5
	print(f"d_loss: {d_loss:.6f}")
	return d_loss

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
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
		# transforms.Lambda(lambda x : x.view(-1)) # flatten the 28x28 image to 1D
	])
	cifar = CIFAR10(args.data_dir, train=True, transform=transform, download=True)
	dataset = DataLoader(dataset=cifar, batch_size=args.batch_size, shuffle=True)

	# model setup
	cvae = CVAE(input_channel=args.input_channel, condition_dim=args.num_classes, latent_dim=args.latent_size).to(device)
	discriminator = Discriminator(in_channel=args.input_channel, condition_dim=args.num_classes).to(device)
	optim_G = torch.optim.Adam(cvae.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
	optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

	real_label = 1.0 # fixed...
	fake_label = 0.0

	# Training
	cvae.train()
	discriminator.train()
	# d_train = True
	for epoch in range(args.epochs):

		# 每隔一个epoch，先冻结判别器，只训练生成器
		# if epoch % 2 == 0:
		# 	d_train = False
		# 	for param in discriminator.parameters():
		# 		param.requires_grad = False
		# else:
		# 	for param in discriminator.parameters():
		# 		param.requires_grad = True

		g_epoch_loss = 0
		d_epoch_loss = 0

		step_counter = 0

		for x, y in dataset:
			x, y = x.to(device), y.to(device)
			c = nn.functional.one_hot(y, num_classes=args.num_classes).float().to(device) # one-hot encoding

			""" update Discriminator """
			if step_counter == 0:
				optim_D.zero_grad()

				real_validity = discriminator(x, c)
				real_labels = torch.full((x.size(0),), real_label, dtype=torch.float, device=device).unsqueeze(-1)
				z = torch.randn(x.size(0), args.latent_size, device=device)
				x_fake = cvae.inference(z, c)
				fake_validity = discriminator(x_fake.detach(), c)  # fixed...
				fake_labels = torch.full((x.size(0),), fake_label, dtype=torch.float, device=device).unsqueeze(-1)

				d_loss = loss_D(real_validity, real_labels, fake_validity, fake_labels)
				d_loss.backward()
				optim_D.step()

				d_epoch_loss += d_loss.item()

			""" update generator """
			for _ in range(args.gd_ratio):
				optim_G.zero_grad()
				x_recon, m, log = cvae(x, c)
				fake_validity = discriminator(x_recon, c) # fixed...
				g_loss = loss_G(x_recon, x, m, log, fake_validity, real_labels, args)
				g_loss.backward()
				optim_G.step()

				g_epoch_loss += g_loss.item()

			# update counter
			step_counter += 1

		# keep the best model parameters according to avg_loss
		g_avg_loss = g_epoch_loss / (len(dataset)*args.gd_ratio)
		d_avg_loss = d_epoch_loss
		# tracker = {"epoch" : None, "criterion" : None}
		if tracker["criterion"] is None or g_avg_loss < tracker["criterion"]:
			tracker["epoch"] = epoch + 1
			tracker["criterion"] = g_avg_loss
			torch.save(cvae.state_dict(), args.model_path)
			pass
		logger.write(f"Epoch {epoch + 1}/{args.epochs}, D_loss: {d_avg_loss:.6f}, G_Loss: {g_avg_loss:.6f}\n")

	# end Training
	logger.write("\n\nTraining completed\n\n")
	logger.write(f"Best Epoch: {tracker['epoch']}, average loss:{tracker['criterion']}\n")
	if os.path.exists(args.model_path):
		logger.write(f"model with the best performance saved to {args.model_path}.")
	# close
	logger.close()