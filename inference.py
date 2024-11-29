import os
import torch
import numpy as np
from model import CVAE
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端

class_dict = {
    "plane": 0,
    "car": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(3, 1, 1)  # 调整形状以匹配张量
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean  # 逆归一化
    return tensor.clamp(0, 1)  # 限制范围在 [0, 1]

def to_uint8(tensor):
    return (tensor * 255).byte()

def generate_images(model, device, text, args):
    digit = class_dict[text] # int
    model.eval()
    with torch.no_grad():
        condition = torch.nn.functional.one_hot(
            torch.tensor([digit] * args.num_classes), num_classes=10
        ).float().to(device)
        z = torch.randn((args.num_classes, args.latent_size)).to(device)  # Random latent vectors
        generated_images = model.inference(z, condition)
        # generated_images = generated_images.view(-1, 28, 28).cpu().numpy()
        # fixed...
        # assert torch.min(generated_images) >= -2 and torch.max(generated_images) <= 2, "Output range mismatch."
        # generated_images = generated_images.cpu().numpy()
        # generated_images = (generated_images.cpu().numpy() + 1) / 2# 将 [-1, 1] 转为 [0, 1]
        # 假设 `generated_images` 是模型的输出，范围为 [-1, 1]
        print(generated_images[0])
        denorm_images = denormalize(generated_images, args.tran_mean, args.tran_std)
        uint8_images = to_uint8(denorm_images)

        #test
        print(uint8_images[0])

        # Plot images
        if args.recon_dir is not None and not os.path.exists(args.recon_dir):
            os.mkdir(args.recon_dir)
        digit_path = os.path.join(args.recon_dir, text)
        if digit_path is not None and not os.path.exists(digit_path):
            os.mkdir(digit_path)

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
            plt.savefig(os.path.join(digit_path, f'#{i + 1}.png'))
            plt.close()  # 保存后再关闭

        print(f"generated images saved to {args.recon_dir}")

def infer(args):
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model setup
    model = CVAE(input_channel=args.input_channel, condition_dim=args.num_classes, latent_dim=args.latent_size).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load(args.model_path))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    model.eval()  # 设置模型为评估模式（如不需训练时）
    print(f"Model loaded from {args.model_path}")

    # 根据输入的文本，生成对应的图片
    text = str(input("请输入要生成图片的类型：")).lower() # 输入对应英文单词
    while text not in class_dict.keys():
        text = input(f"无效类别，请输入以下之一: {list(class_dict.keys())}\n")
    generate_images(model, device, text=text, args=args)

    pass