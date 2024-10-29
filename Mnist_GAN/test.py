import cv2
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt

from network import Generator,Discriminator

G = Generator(input_size=100,output_size=28*28)
D = Discriminator(input_size=28*28)

G.load_state_dict(torch.load("G_model.pt"))
D.load_state_dict(torch.load("D_model.pt"))

with torch.no_grad():
    noise = torch.randn(12,100)
    fake_img = G(noise)
    fake_img = fake_img.view(12,28,28)

    fig = plt.figure()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(fake_img[i], cmap="gray", interpolation="none")
        plt.xticks([])
        plt.yticks([])
    plt.show()
