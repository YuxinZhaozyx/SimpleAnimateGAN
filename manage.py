import torch

class Manager(object):
    
    def __init__(self, netG, netD=None, optimizerG=None, optimizerD=None, criterion=None, device='cpu'):
        self.netG = netG
        self.netD = netD 
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.device = device

    def train(self, dataloader, epoch):
        if not (self.netG and self.netD and self.optimizerD and self.optimizerG and self.criterion):
            raise Exception("Manager has not enough init parameters to train")

        for index, images in enumerate(dataloader):
            batch_size = images.size(0)

            """ Update D network: maximize log(D(x)) + log(1 - D(G(z))) """
            self.netD.zero_grad()

            # train with all real batch
            # real label
            labels = torch.full((batch_size,), 1, device=self.device)
            outputs = self.netD(images.to(self.device)).view(-1)
            errD_real = self.criterion(outputs, labels)
            errD_real.backward()
            D_x = outputs.mean().item()

            # train with all fake batch
            # Generate batch of latent vectors
            noises = torch.randn(batch_size, 100, device=self.device)
            # Generate fake image batch with G
            fakes = self.netG(noises)
            # fake label
            labels.fill_(0)
            outputs = self.netD(fakes).view(-1)
            errD_fake = self.criterion(outputs, labels)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = outputs.mean().item()

            # add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # update D's parameters
            self.optimizerD.step()


            """ Update G network: maximize log(D(G(z))) """
            self.netG.zero_grad()

            # real label
            labels.fill_(1)
            outputs = self.netD(fakes).view(-1)
            errG = self.criterion(outputs, labels)
            errG.backward()
            D_G_z2 = outputs.mean().item()

            # update G's parameters
            self.optimizerG.step()

            # output training status
            print('Epoch: %d [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, index + 1, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))



            
            

            

