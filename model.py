import torch
import torch.nn as nn 
import torch.nn.functional as F 

# DCGAN
class Generator(nn.Module):
    def __init__(self, input_dim = 100):
        super().__init__()
        
        self.main = nn.Sequential(
            # input_dim x 1 x 1  to  512 x 4 x 4
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 512 x 4 x 4  to  256 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256 x 8 x 8  to  128 x 16 x 16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 16 x 16  to  64 x 32 x 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 32 x 32  to  3 x 64 x 64
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.__weight_init()

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.main(x)
        return x

    def __weight_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 3 x 64 x 64  to  64 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32  to  128 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16  to  256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8  to  512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4  to  1 x 1 x 1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.__weight_init()

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1)
        return x

    def __weight_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    """ Test for Generator and Discriminator """
    import time

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define netG and netD
    netG = Generator()
    netD = Discriminator()
    
    # init netG and netD
    netG.to(device)
    netD.to(device)
    
    # Test for Generator
    start_time = time.time()
    x = torch.rand(128, 100).to(device)
    y = netG(x)
    end_time = time.time()
    print("generator cost time: ", end_time - start_time)
    print(y.cpu().size())

    # Test for Discriminator
    start_time = time.time()
    x = torch.rand(128, 3, 64, 64).to(device)
    y = netD(x)
    end_time = time.time()
    print("discriminator cost time: ", end_time - start_time)
    print(y.cpu().size())
    


        