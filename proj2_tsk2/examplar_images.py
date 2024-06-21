import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations
transforms_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transforms_aug1 = transforms.Compose([
    transforms.AutoAugment(),  # AutoAugment
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transforms_aug2 = transforms.Compose([
    transforms.RandomRotation(30),  # Random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# Get an example image from each dataset
mnist_image, _ = mnist_dataset[0]
cifar10_image, _ = cifar10_dataset[0]

# Apply transformations
mnist_no_aug = transforms_no_aug(mnist_image)
mnist_aug1 = transforms_aug1(mnist_image)
mnist_aug2 = transforms_aug2(mnist_image)

cifar10_no_aug = transforms_no_aug(cifar10_image)
cifar10_aug1 = transforms_aug1(cifar10_image)
cifar10_aug2 = transforms_aug2(cifar10_image)

# Helper function to unnormalize and convert tensor to numpy array
def unnormalize(tensor, mean=0.5, std=0.5):
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.numpy()

def show_images(images, titles, cmap=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        axs[i].imshow(image, cmap=cmap)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.show()

# Unnormalize and convert to numpy arrays
mnist_images = [unnormalize(mnist_no_aug.squeeze()), 
                unnormalize(mnist_aug1.squeeze()), 
                unnormalize(mnist_aug2.squeeze())]

cifar10_images = [unnormalize(cifar10_no_aug.permute(1, 2, 0)), 
                  unnormalize(cifar10_aug1.permute(1, 2, 0)), 
                  unnormalize(cifar10_aug2.permute(1, 2, 0))]

# Display images
show_images(mnist_images, ['MNIST No Augmentation', 'MNIST AutoAugment', 'MNIST RandomRotation'], cmap='gray')
show_images(cifar10_images, ['CIFAR10 No Augmentation', 'CIFAR10 AutoAugment', 'CIFAR10 RandomRotation'])