import torch
import torchvision
import torchvision.transforms as transforms

def fashion_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=transform, download=True
    )
    valid_dataset = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False)
    
    return train_loader, valid_loader