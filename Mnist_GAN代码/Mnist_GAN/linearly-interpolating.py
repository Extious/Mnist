import torch
from matplotlib import pyplot as plt

from network import Generator, Discriminator

# md_img = Image.open("./md_img/0-tensor(5).png").convert("L")
# plt.figure()
# plt.imshow(md_img,cmap='gray')
# plt.show()
# 43 9583
torch.manual_seed(958431)
noise1 = torch.randn(1, 100)
noise2 = torch.randn(1, 100)


G = Generator(input_size=100, output_size=28 * 28)
D = Discriminator(input_size=28 * 28)
G.load_state_dict(torch.load("G_model.pt"))
D.load_state_dict(torch.load("D_model.pt"))

origin_img1 = G(noise1).view(1, 28, 28)
origin_img2 = G(noise2).view(1, 28, 28)


def linear_interpolate(v1, v2, t):
    if t == -1:
        return v1
    if t == 10:
        return v2
    img = v1.clone()
    img[:, 0:t * 10 + 10] = v2[:, 0:t * 10 + 10]
    return img
# print(noise1,noise2)
for i in range(-1, 10):
    noise = linear_interpolate(noise1, noise2, i)
    with torch.no_grad():
        fake_img = G(noise)
        fake_img = fake_img.view(28, 28)
    # print(noise)
    plt.subplot(1, 11, i + 2)
    plt.imshow(fake_img, cmap="grey")
    plt.xticks([])
    plt.yticks([])

plt.show()


