# from torchvision.datasets import STL10
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloader(args):
	# data preparations
	transform = transforms.Compose([
		transforms.Resize((args.img_size, args.img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=args.tran_mean, std=args.tran_std),
	])
	# stl = STL10(args.data_dir, split="train", transform=transform, download=True)
	data = datasets.ImageFolder(root='data/images', transform=transform)
	dataset = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, drop_last=True)

	class_idx = data.class_to_idx # {类名: 类编号}
	return dataset, class_idx

def test(args):
	dataset, class_idx = get_dataloader(args=args)
	cnt = 0
	print(class_idx)
	for x, y in dataset:
		if cnt == 10:
			break
		# print(y)
		print(x)
		cnt += 1
		pass