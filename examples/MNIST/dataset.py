import cv2
import torchvision
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class MNISTSearchDataset(torchvision.datasets.MNIST):
    def __init__(self, root="~/data/mnist", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            im_3d = np.repeat(image[:, :, np.newaxis], 3, 2)
            transformed = self.transform(image=im_3d.numpy())
            image = transformed["image"]

        return image, label
