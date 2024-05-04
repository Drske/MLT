import torch
import torchvision
import torchvision.transforms as transforms

def cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    return train_loader, test_loader