import torch
from torchvision import transforms
from model import Generator
import os

if __name__ == '__main__':
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    netG = Generator()
    netG.to(device)

    # generate
    generated_save_path = './generated'
    num_generated_image = 100

    # weight
    weight_path = './log/G-weight-00001000.log'
    netG.load_state_dict(torch.load(weight_path))

    random_num = int(torch.rand(1).item() * 1000)
    noises = torch.randn(num_generated_image, 100, device=device)

    with torch.no_grad():
            images = netG(noises).cpu()
            for i in range(num_generated_image):
                image = transforms.ToPILImage()(images[i])
                image.save(os.path.join(generated_save_path, '{:0>3}-{:0>4}.jpg'.format(random_num, i+1)))
