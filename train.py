import torch
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from datasets import AnimateDataset
from manage import Manager 
from model import Generator, Discriminator
import os

if __name__ == '__main__':
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # dataset
    dataset_path = 'E:/Dataset/AnimateDataset/faces'
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    dataset = AnimateDataset(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # model
    netG = Generator()
    netD = Discriminator()
    netG.to(device)
    netD.to(device)

    # optimizer
    optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    schedulerG = lr_scheduler.ExponentialLR(optimizerG, gamma=0.9)
    schedulerD = lr_scheduler.ExponentialLR(optimizerD, gamma=0.9)
    
    # criterion
    criterion = torch.nn.BCELoss()

    # manager
    manager = Manager(netG, netD, optimizerG, optimizerD, criterion, device)

    # generate
    generated_save_path = './generated'
    num_generated_image = 5

    # epochs
    epochs = 500

    # load
    start_epoch = 1
    if os.path.exists('./log/lastest_checkpoint.log'):
        lastest_checkpoint = torch.load('./log/lastest_checkpoint.log')
        lastest_saved_epoch = lastest_checkpoint['epoch']
        netG.load_state_dict(torch.load('./log/G-weight-{:0>8}.log'.format(lastest_saved_epoch)))
        netD.load_state_dict(torch.load('./log/D-weight-{:0>8}.log'.format(lastest_saved_epoch)))
        optimizerG.load_state_dict(lastest_checkpoint['optimizerG'])
        optimizerD.load_state_dict(lastest_checkpoint['optimizerD'])
        start_epoch = lastest_saved_epoch + 1

    for epoch in range(start_epoch, epochs+1):
        print("\n---- EPOCH %d ----\n" % epoch)
        # train one epoch
        manager.train(dataloader, epoch)
        schedulerG.step()
        schedulerD.step()

        if epoch % 10 == 0:
            # save weights
            torch.save(netG.state_dict(), './log/G-weight-{:0>8}.log'.format(epoch))
            torch.save(netD.state_dict(), './log/D-weight-{:0>8}.log'.format(epoch))
            lastest_checkpoint = {'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict(), 'epoch': epoch}
            torch.save(lastest_checkpoint, './log/lastest_checkpoint.log')

        # save some generated image
        noises = torch.randn(num_generated_image, 100, device=device)
        with torch.no_grad():
            images = netG(noises).cpu()
            for i in range(num_generated_image):
                image = transforms.ToPILImage()(images[i])
                image.save(os.path.join(generated_save_path, '{:0>8}_{:0>4}.jpg'.format(epoch, i+1)))


        
