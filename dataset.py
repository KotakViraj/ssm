import torch
import torchvision
import torchvision.transforms as transforms

# MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()),
        ])

    train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True,)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False,)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

Datasets = {"mnist": create_mnist_dataset}