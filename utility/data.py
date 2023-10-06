import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Cifar100:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):
        mean, std = self._get_statistics()

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        subset_train=list(range(0,50000)) 
        subset_test1 = list(range(0, 1000)) # subsample test data for training the mask
        subset_test2 = list(range(1000,10000)) # evaluation set
        testset1 = torch.utils.data.Subset(test_set, subset_test1)
        testset2 = torch.utils.data.Subset(test_set,subset_test2)
        trainset=torch.utils.data.Subset(train_set,subset_train)
        self.train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=threads)
       # self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.test2 =  torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False, num_workers=threads)
        #self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        #self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar10:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):

        mean, std = self._get_statistics()
        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        subset_train=list(range(0,50000))  # subsample training data to shorten running time in the code testing stage
        subset_test1 = list(range(0, 1000)) # subsample test data for training the mask
        subset_test2 = list(range(1000,10000)) # evaluation set
        testset1 = torch.utils.data.Subset(test_set, subset_test1)
        testset2 = torch.utils.data.Subset(test_set,subset_test2)
        trainset=torch.utils.data.Subset(train_set,subset_train)
        self.train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=threads)
       # self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.test2 =  torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False, num_workers=threads)
    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class FashionMNIST:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):
        mean, std = self._get_statistics()

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
        subset_train=list(range(0,50000)) 
        subset_test1 = list(range(0, 1000)) # subsample test data for training the mask
        subset_test2 = list(range(1000,10000)) # evaluation set
        testset1 = torch.utils.data.Subset(test_set, subset_test1)
        testset2 = torch.utils.data.Subset(test_set,subset_test2)
        trainset=torch.utils.data.Subset(train_set,subset_train)
        self.train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=threads)
       # self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.test2 =  torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False, num_workers=threads)
    def _get_statistics(self):
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class MNIST:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):
        mean, std = self._get_statistics()

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        subset_train=list(range(0,50000)) 
        subset_test1 = list(range(0, 1000)) # subsample test data for training the mask
        subset_test2 = list(range(1000,10000)) # evaluation set
        testset1 = torch.utils.data.Subset(test_set, subset_test1)
        testset2 = torch.utils.data.Subset(test_set,subset_test2)
        trainset=torch.utils.data.Subset(train_set,subset_train)
        self.train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=threads)
       # self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test1 = torch.utils.data.DataLoader(testset1, batch_size=100, shuffle=True, num_workers=threads)
        self.test2 =  torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False, num_workers=threads)
    def _get_statistics(self):
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
