import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from network import Generator,Discriminator
from mnist_dataset import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def D_train(image):
    real_label = torch.ones(num_img).unsqueeze(1)
    fake_label = torch.zeros(num_img).unsqueeze(1)

    # 计算真实图片判别误差
    real_out = D(image)
    d_loss_real = criterion(real_out, real_label)

    noise = torch.randn(num_img, 100)
    fake_img = G(noise)
    # 计算生成图片判别误差
    fake_out = D(fake_img)
    d_loss_fake = criterion(fake_out, fake_label)

    # 判别器训练
    D_loss = d_loss_real + d_loss_fake
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()
    return D_loss.data.item()


def G_train(x):
    real_label = torch.ones(num_img).unsqueeze(1)
    # 生成器训练
    noise = torch.randn(num_img, 100)
    fake_img = G(noise)
    output = D(fake_img)
    G_loss = criterion(output, real_label)
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()


if __name__ == "__main__":
    batch_size = 128
    lr = 0.0003
    num_epochs = 50

    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    G = Generator(input_size=100,output_size=28*28).to(device)
    D = Discriminator(input_size=28*28).to(device)
    G.load_state_dict(torch.load("G_model.pt"))
    D.load_state_dict(torch.load("D_model.pt"))

    criterion = nn.BCELoss()
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)


    G_loss_plt = []
    D_loss_plt = []

    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(train_loader):
            num_img = image.size(0)
            image = image.view(num_img, -1)
            D_loss = D_train(image)
            G_loss = G_train(image)

            G_loss_plt.append(G_loss)
            D_loss_plt.append(D_loss)

            if (i + 1) % 50 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss_D: %.4f, Loss_G: %.4f, AVG_lossD: %.4f, AVG_lossG: %.4f' % (
                    epoch + 1, num_epochs, i + 1, len(train_loader),
                    D_loss, G_loss, torch.mean(torch.FloatTensor(D_loss_plt)), torch.mean(torch.FloatTensor(G_loss_plt))))



    torch.save(G.state_dict(), "G_model.pt")
    torch.save(D.state_dict(), "D_model.pt")


    # plt.plot(G_loss_plt)
    # plt.plot(D_loss_plt)
    # plt.xlabel("Iter")
    # plt.ylabel("loss")
    # plt.show()
