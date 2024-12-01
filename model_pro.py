import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Encoder(nn.Module):
	"""
	:class param
	input_channel: 输入图片的通道数
	condition_dim: 条件独热编码的维度
	latent_dim: 潜在向量的维度

	:forward param
	x: 输入图片数据
	c: 条件独热编码向量

	:return
	m: 重采样之后的均值
	log: 重采样之后的方差的对数
	"""
	def __init__(self, input_channel, condition_dim, latent_dim):
		super(Encoder, self).__init__()

		self.input_channel = input_channel + condition_dim

		self.resnet = models.resnet18(pretrained=True)
		self.resnet.conv1= nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
		# pool : (bs, 512, 4, 4) -> (bs, 512, 1, 1)
		self.resnet.avgpool = nn.AvgPool2d(4, 1, 0)
		self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

		self.mean_layer = nn.Linear(512, latent_dim)
		self.log_layer = nn.Linear(512, latent_dim)

	# def reparameterize(self, x):
	# 	m = self.mean_layer(x) # (bs, latent_dim)
	# 	log = self.log_layer(x) # (bs, latent_dim)
	# 	std = torch.exp(0.5 * log) + 1e-6
	# 	eps = torch.randn_like(std)
	#
	# 	z = m + eps * std
	# 	# kld = -0.5 * torch.sum(1 + log - m.pow(2) - log.exp())
	#
	# 	return z, m, log

	def forward(self, x, c):
		# c: [batch_size, num_classes]
		# -> [batch_size, num_classes, 1, 1]
		# -> [batch_size, num_classes, x.size(2), x.size(3)]
		c = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))

		x = torch.cat([x, c], dim=1)  # 拼接图片数据向量和条件对应的独热编码
		x = self.resnet(x)
		x = x.squeeze()  # (bs, 512)

		# 重参数化
		m = self.mean_layer(x) # (bs, latent_dim)
		log = self.log_layer(x) # (bs, latent_dim)

		return m, log


class Decoder(nn.Module):
	"""
	:class param
	latent_dim: 潜在向量的维度
	condition_dim: 条件独热编码的维度
	input_channel: 输入图片的通道数

	:forward param
	z: 潜在向量
	c: 条件独热编码向量

	:return
	x_recon: 重建数据
	z: 潜在向量
	"""
	def __init__(self, latent_dim, condition_dim, input_channel):
		super(Decoder, self).__init__()

		self.latent_dim = latent_dim + condition_dim
		self.dec_input = nn.Linear(self.latent_dim, latent_dim)
		self.dec_mlp = nn.Sequential(
			nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False), # (bs, 512, 4, 4)
			nn.ReLU(),

			nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False), # (bs, 384, 8, 8)
			nn.BatchNorm2d(384),
			nn.ReLU(),

			nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False), # (bs, 192, 16, 16)
			nn.BatchNorm2d(192),
			nn.ReLU(),

			nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False), # (bs, 96, 32, 32)
			nn.BatchNorm2d(96),
			nn.ReLU(),

			nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),  # (bs, 64, 64, 64)
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),

			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # (bs, 3, 128, 128)
			nn.Tanh(), # [-1, 1]
		)

	def forward(self, z, c):
		z = torch.cat([z, c], dim=1)  # 拼接潜在向量和条件对应的独热编码
		z = self.dec_input(z) # ->(bs, 200)
		z0 = z.size(0) # bs
		z1 = z.size(1) # 200
		out = z.view(z0, z1, 1, 1)
		x_recon = self.dec_mlp(out)

		return x_recon


class CVAE(nn.Module):
	def __init__(self, input_channel, condition_dim, latent_dim):
		# Encoder & Decoder 初始化
		super(CVAE, self).__init__()
		self.enc = Encoder(input_channel, condition_dim, latent_dim)
		self.dec = Decoder(latent_dim, condition_dim, input_channel)

	def reparameterize(self, mean, log):
		std = torch.exp(0.5 * log) + 1e-6
		eps = torch.randn_like(std)
		return mean + std * eps

	def inference(self, z, c):
		x_recon = self.dec(z, c)
		return x_recon

	def forward(self, x, c):
		m, log = self.enc(x, c)
		z = self.reparameterize(m, log)
		x_recon = self.dec(z, c)

		return x_recon, m, log

class Discriminator(nn.Module):
	def __init__(self, in_channel, condition_dim):
		super(Discriminator, self).__init__()
		self.in_channel = in_channel + condition_dim # 3+5
		self.mlp = nn.Sequential(
			nn.Conv2d(self.in_channel, 64, 4, 2, 1, bias=False), # (bs, 64, 64, 64)
			nn.LeakyReLU(0.2),

			nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (bs, 128, 32, 32)
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (bs, 256, 16, 16)
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (bs, 512, 8, 8)
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			nn.Conv2d(512, 512, 4, 2, 1, bias=False),  # (bs, 512, 4, 4)
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
		)

		self.last_conv = nn.Sequential(
			nn.Conv2d(512, 1, 4, 1, 0),
			nn.Sigmoid()
		)

	def forward(self, x, c):
		assert x.dim() == 4, "Input x must be 4D tensor (batch, channel, height, width)"
		assert c.dim() == 2, "Condition c must be 2D tensor (batch, condition_dim)"

		c = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
		x = torch.cat([x, c], dim=1)  # (bs, in, 128, 128)
		fm = self.mlp(x) # (bs, 512, 3, 3)
		x = self.last_conv(fm) # (bs, 1, 1, 1)
		# fm = F.avg_pool2d(fm, 3, 1, 0) # (bs, 512, 1, 1)
		# fm_pooled = F.avg_pool2d(fm, 3, 1, 0)  # (bs, 512, 1, 1)

		# fixing...
		return x.squeeze(dim=-1).squeeze(dim=-1)  # (bs)
		# return x.squeeze(), fm_pooled.squeeze()
		# return x.squeeze(dim=-1).squeeze(dim=-1), f_d.squeeze()
