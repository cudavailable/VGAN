import os
import torch
from torch import nn
# from torchvision.datasets import CIFAR10, STL10

from model_pro import CVAE, Discriminator
from logger import Logger
from dataset import get_dataloader

from matplotlib import pyplot as plt
from inference import denormalize, to_uint8
from PIL import Image

def test_reverse(x, y, args):
	denorm_images = denormalize(x[0:3], mean=args.tran_mean, std=args.tran_std)
	uint8_images = to_uint8(denorm_images)
	print(uint8_images[0])
	plt.figure()
	for i, img in enumerate(uint8_images):
		# 如果生成的是 1xHxW，调整为 HxWxC
		# if img.shape[0] == 3:  # 检查是否为 (3, H, W)
		# 	img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, C)
		#
		# img = (img * 255).astype(np.uint8)  # 转换为 [0, 255]
		img = img.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
		img = Image.fromarray(img)

		plt.imshow(img)
		plt.axis('off')
		plt.savefig(f'#{y[i]}.png')
		plt.close()  # 保存后再关闭

	print(f"generated images saved to {args.recon_dir}")

# loss function for cvae
def loss_G(x_recon, x, mean, log, fake_validity, real_labels, args):
	recon_loss = nn.MSELoss()(x_recon, x)
	kl_div = -0.5 * torch.sum(1 + log - mean.pow(2) - log.exp())

	#test
	# print(f"fake_validity shape: {fake_validity.shape}, real_labels shape: {real_labels.shape}")

	adv_loss = nn.BCELoss()(fake_validity, real_labels)

	recon_loss = recon_loss * args.w_recon
	kl_div = kl_div * args.w_kl
	adv_loss = adv_loss * args.w_adv

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
	global real_labels
	if args.log_dir is not None and not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	logger = Logger(os.path.join(args.log_dir, args.log_path))

	# keep the best model parameters according to avg_loss
	# tracker = {"epoch" : None, "criterion" : None}

	# device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.write(f"we're using :: {device}\n\n")

	# data preparations
	# transform = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(mean=args.tran_mean, std=args.tran_std),
	# ])
	# cifar = CIFAR10(args.data_dir, train=True, transform=transform, download=True)
	# dataset = DataLoader(dataset=cifar, batch_size=args.batch_size, shuffle=True, drop_last=True)
	dataset, class_idx  = get_dataloader(args)

	# model setup
	cvae = CVAE(input_channel=args.input_channel, condition_dim=args.num_classes, latent_dim=args.latent_size).to(device)
	discriminator = Discriminator(in_channel=args.input_channel, condition_dim=args.num_classes).to(device)
	optim_G = torch.optim.Adam(cvae.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
	optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

	if args.preTrain == True:
		cvae.load_state_dict(torch.load("cvae_model.pth"))
		# 打印模型的一个参数，检查加载是否成功
		for name, param in cvae.named_parameters():
			print(f"{name}: {len(param.data)}")
			break
		logger.write(f"pretrained model: {args.model_path}\n\n")
		

	real_label = 1.0 # fixed...
	fake_label = 0.0

	# Training
	cvae.train()
	discriminator.train()
	for epoch in range(args.epochs):

		g_epoch_loss = 0
		d_epoch_loss = 0

		# step_counter = 0

		for x, y in dataset:

			x, y = x.to(device), y.to(device)
			c = nn.functional.one_hot(y, num_classes=args.num_classes).float().to(device) # one-hot encoding

			real_labels = torch.full((x.size(0),), real_label, dtype=torch.float, device=device).unsqueeze(-1)
			fake_labels = torch.full((x.size(0),), fake_label, dtype=torch.float, device=device).unsqueeze(-1)

			## test
			# print(x[0])
			# print("")
			# test_reverse(x, y, args)
			# exit(0)

			# if step_counter == 0:
			optim_D.zero_grad()
			optim_G.zero_grad()

			""" 数据准备1 """
			# 真实图像进行判别
			vd_r = discriminator(x, c)

			""" 数据准备2 """
			# 生成潜在向量
			z = torch.randn(x.size(0), args.latent_size, device=device)
			# 潜在向量z生成图像，并判别
			x_p = cvae.inference(z, c)
			vd_p = discriminator(x_p.detach(), c)  # fixed...


			""" update Discriminator """
			d_loss = loss_D(vd_r, real_labels, vd_p, fake_labels)
			d_loss.backward(retain_graph=True)
			optim_D.step()

			d_epoch_loss += d_loss.item()

			""" update generator """
			for _ in range(args.gd_ratio):
				""" 数据准备3 """
				x_f, m, log = cvae(x, c)
				vd_f = discriminator(x_f, c)  # fixed...
				g_loss = loss_G(x_f, x, m, log, vd_f, real_labels, args)
				g_loss.backward()
				optim_G.step()

				g_epoch_loss += g_loss.item()

			# update counter
			# step_counter += 1

		# keep the best model parameters according to avg_loss
		g_avg_loss = g_epoch_loss / (len(dataset) * args.gd_ratio)
		d_avg_loss = d_epoch_loss / len(dataset)

		torch.save(cvae.state_dict(), args.model_path)
		torch.save(discriminator.state_dict(), args.D_path)
		logger.write(f"Epoch {epoch + 1}/{args.epochs}, D_loss: {d_avg_loss:.6f}, G_Loss: {g_avg_loss:.6f}\n")

	# end Training
	logger.write("\n\nTraining completed\n\n")
	if os.path.exists(args.model_path):
		logger.write(f"model with the best performance saved to {args.model_path}.\n")
	# close
	logger.close()