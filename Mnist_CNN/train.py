import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from LeNet_5 import LeNet_5
from mnist_dataset import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    batch_size = 256
    lr = 0.001
    num_epochs = 30

    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet_5().to(device)
    model.load_state_dict(torch.load("model.pt"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    accuracy = []

    for epoch in range(num_epochs):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f,' % (
                    epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        model.eval()
        correct = 0
        total = 0
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            probs, predictions = torch.max(output, 1)
            total += label.size(0)
            correct += (predictions == label).sum().item()

        accuracy.append(correct / total)
        print("Accuracy: %.4f" % (correct / total))

    torch.save(model.state_dict(), "model.pt")

    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
