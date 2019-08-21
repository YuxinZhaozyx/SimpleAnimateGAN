import torch
import os
from visdom import Visdom
import numpy as np
from sklearn import metrics

# init Visdom instance
log_filename = './log/simple-animate-GAN.log'
vis = Visdom(env="simple-animate-GAN", log_to_filename=log_filename)
# ensure the Visdom server is online
assert(vis.check_connection())
# vis.replay_log(log_filename)


class LossLogger:
    def __init__(self, win='loss'):
        self.to_reset = False
        self.win = win
    
    # set reset and the window will be cleared and updated at next time call 'add' method
    def reset(self):
        self.to_reset = True
        print("[info] reset loss window \"" + self.win + "\"")
    
    # append a new loss
    def add(self, epoch, loss):
        # check whether need to reset
        if self.to_reset:
            update = None
            self.to_reset = False
        else:
            update = 'append'

        # draw loss curve
        vis.line(
            X = np.array([epoch]),
            Y = np.array([loss]),
            win = self.win,
            opts = dict(
                title = 'loss',
                ytickmin = 0.0,
                xlabel = 'epoch',
                ylabel = 'loss'
            ),
            update = update
        )
        

# save and laod State
class StateLogger:

    def __init__(self):
        self.path = None
        self.model = None
        self.optimizer = None

    def __init__(self, path, model, optimizer):
        self.path = path
        self.model = model
        self.optimizer = optimizer
    
    # save state
    def save(self, epoch):
        self.save(self.path, self.model, self.optimizer, self.epoch)

    # load state
    # return epoch
    def load(self, epoch):
        return self.load(self.path, self.model, self.optimizer)

    # save state
    @staticmethod
    def save(path, model, optimizer, epoch):
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(checkpoint, path)
        print("[save] epoch: {}".format(epoch))

    # load state
    @staticmethod
    def load(path, model, optimizer):
        epoch = 0
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            print("[load] epoch: {}".format(epoch))
        return epoch


if __name__ == '__main__':
    import sklearn
    import math

    # test LossLogger and Accuracy Logger
    loss_logger = LossLogger()
    acc_logger = AccuracyLogger()

    loss_logger.reset()
    acc_logger.reset()

    loss = 2000

    for epoch in range(2, 100, 2):
        loss_logger.add(epoch, loss)
        loss /= 2

        acc = 1 / (1 + math.exp(-epoch))
        acc_logger.add(epoch, acc)
    