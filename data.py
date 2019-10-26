from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

_SVHN_TARGET_TRANSFORMS = [
    transforms.Lambda(lambda y: y % 10)
]

_CELEBA_TRAIN_TRANSFORMS_64 = [
    transforms.CenterCrop(140),
    transforms.Scale((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

_CELEBA_TEST_TRANSFORMS_64 = [
    transforms.CenterCrop(140),
    transforms.Scale((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


class MNISTwIDX(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class FASHIONwIDX(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class SVHNwIDX(datasets.SVHN):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class SVHNwIDX(datasets.SVHN):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class CelebAwIDX(ImageFolder):
    def __init__(self, root, transform=None):
        super(CelebAwIDX, self).__init__(root, transform)
        self.indices = range(len(self))
        print(root)
        print(len(self))

    def __getitem__(self, index1):
        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, 0, index1




TRAIN_DATASETS = {
    'mnist': MNISTwIDX(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'fashion': FASHIONwIDX(
        './datasets/fashion', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'svhn': SVHNwIDX(
        './datasets/svhn', split='train', download=True,
        transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),
    'CelebA': CelebAwIDX(
        '/home/ubuntu/datasets/CelebA/img_align_celeba/train',
        transform=transforms.Compose(_CELEBA_TRAIN_TRANSFORMS_64),
    ),
}


TEST_DATASETS = {
    'mnist': MNISTwIDX(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'fashion': FASHIONwIDX(
        './datasets/fashion', train=False, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'svhn': SVHNwIDX(
        './datasets/svhn', split='test', download=True,
        transform=transforms.Compose(_SVHN_TEST_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),
    'CelebA':CelebAwIDX(
        '/home/ubuntu/datasets/CelebA/img_align_celeba/test',
        transform=transforms.Compose(_CELEBA_TRAIN_TRANSFORMS_64),
    ),
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'fashion': {'size': 32, 'channels': 1, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'CelebA': {'size': 64, 'channels': 3, 'classes': 10},
}
