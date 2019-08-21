from torch.utils.data import Dataset
import os
from PIL import Image

class AnimateDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = [os.path.join(path, file) for file in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """ get a PIL image """
        path = self.image_paths[index]
        image = Image.open(path, "r")
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == '__main__':
    """ Test for AnimateDataset """
    path = 'E:/Dataset/AnimateDataset/faces'
    dataset = AnimateDataset(path)
    for i in range(3):
        image = dataset[i]
        image.show()
