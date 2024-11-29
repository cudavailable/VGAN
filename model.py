import torch
import torch.nn as nn

class Encoder(nn.Module):
	"""
	:class param
	input_channel: 输入图片的通道数
	condition_dim: 条件独热编码的维度
	latent_dim: 潜在向量的维度

	:forward param
	x: 输入图片数据[bs, 3, 32, 32]
	c: 条件独热编码向量

	:return
	m: 重采样之后的均值
	log: 重采样之后的方差的对数
	"""
	def __init__(self, input_channel, condition_dim, latent_dim):
		super(Encoder, self).__init__()


		self.input_channel = input_channel + condition_dim  # 3+10
		self.enc_mlp = nn.Sequential(
			nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),  # [bs, 64, 16, 16]
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),  # [bs, 128, 8, 8]
			nn.Flatten(),  # [bs, (128*8*8)]
		)
		self.mean_layer = nn.Linear(128 * 8 * 8, latent_dim)
		self.log_layer = nn.Linear(128 * 8 * 8, latent_dim)

	def forward(self, x, c):
		# [batch_size, num_classes]
		# -> [batch_size, num_classes, 1, 1]
		# -> [batch_size, num_classes, x.size(2), x.size(3)]
		c = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))

		x = torch.cat([x, c], dim=1)  # 拼接图片数据向量和条件对应的独热编码
		z = self.enc_mlp(x)

		# 重参数化
		m = self.mean_layer(z)
		log = self.log_layer(z)

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
		self.dec_input = nn.Linear(self.latent_dim, 128 * 8 * 8)
		self.dec_mlp = nn.Sequential(
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=64, out_channels=input_channel, kernel_size=4, stride=2, padding=1),
			nn.Tanh(),  # size: (bs, 3, 32, 32) value: [0, 1]
		)

	def forward(self, z, c):
		z = torch.cat([z, c], dim=1)  # 拼接潜在向量和条件对应的独热编码
		z = self.dec_input(z).view(-1, 128, 8, 8)
		out = self.dec_mlp(z)
		x_recon = out * 2.065 + 0.075

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
		self.in_channel = in_channel + condition_dim # 3+10
		self.mlp = nn.Sequential(
			nn.Conv2d(in_channels=self.in_channel, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2),  # (bs, 64, 16, 16)
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2),  # (bs, 128, 8, 8)
			nn.Flatten(),
			nn.Linear(128*8*8, 1),
			nn.Sigmoid(),
		)

	def forward(self, x, c):
		assert x.dim() == 4, "Input x must be 4D tensor (batch, channel, height, width)"
		assert c.dim() == 2, "Condition c must be 2D tensor (batch, condition_dim)"

		c = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
		x = torch.cat([x, c], dim=1)
		x = self.mlp(x)

		return x
