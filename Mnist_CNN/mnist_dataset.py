import torchvision
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST("./data/", train=True,
                                               download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data/", train=False,
                                               download=False, transform=transform)
    return train_dataset,test_dataset


if __name__ == "__main__":
    train_dataset,_ = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for image,label in train_loader:
        print(image.shape,label)
        break

    fig = plt.figure()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(train_dataset.train_data[i],cmap="gray",interpolation="none")
        plt.title("Label:{}".format(train_dataset.train_labels[i]))
        # plt.xticks([])
        # plt.yticks([])
    plt.show()
    print(train_dataset.train_data[0].shape)
