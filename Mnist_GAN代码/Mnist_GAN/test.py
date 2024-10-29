import cv2
import numpy as np
from PIL import Image
import torchvision.transforms
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from network import Generator,Discriminator

G = Generator(input_size=100,output_size=28*28)
D = Discriminator(input_size=28*28)

G.load_state_dict(torch.load("G_model.pt"))
D.load_state_dict(torch.load("D_model.pt"))

# with torch.no_grad():
#     noise = torch.randn(12,100)
#     fake_img = G(noise)
#     fake_img = fake_img.view(12,28,28)
#
#     fig = plt.figure()
#     for i in range(12):
#         plt.subplot(3,4,i+1)
#         plt.tight_layout()
#         plt.imshow(fake_img[i], cmap="gray", interpolation="none")
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()


def load_dataset():
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST("./data/", train=True,
                                               download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data/", train=False,
                                               download=False, transform=transform)
    return train_dataset,test_dataset

train_dataset,_ = load_dataset()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
noise = torch.randn(1,100)
fake_img = G(noise)
print(fake_img.shape)

with torch.no_grad():
    min_distance = float('inf')
    closest_image_feature = None
    for i,(images,label) in enumerate(train_loader):
        for j in range(images.shape[0]):
            img = images[j].view(-1).unsqueeze(0)
            current_distance = torch.dist(fake_img,img,2)

            if current_distance < min_distance:
                min_distance = current_distance
                closest_image_feature = img

    fake_img = fake_img.resize(1,28,28)
    closest_image_feature = closest_image_feature.resize(1,28,28)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(fake_img.squeeze(), cmap='gray')
    plt.title('Generate image')
    plt.subplot(1,2,2)
    plt.imshow(closest_image_feature.squeeze(), cmap='gray')
    plt.title('True image')
    plt.show()