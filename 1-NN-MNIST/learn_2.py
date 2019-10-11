'''
Main background for this found here:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
'''
import os
import sys
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class ConvNet(torch.nn.Module):
    '''
    This is our custom model that we are building.  For now, it is copied
    from the tutorial.
    '''

    def __init__(self, device=torch.device("cpu")):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        self.device = device
        if "cuda" in str(device):
            self.cuda(device)

    def forward(self, x):
        '''
        This outlines how the data flows through this NN.
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def setup(batch_size=64, dataset_folder="./datasets"):
    '''
    This function just uses pre-built features of pytorch and torchvision to
    get all the MNIST data and return them as generators that can be looped
    through.  One thing of note is that the DataLoader will iterate through
    batch sizes of the data!
    '''
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    # transforms to apply to the data
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_folder, download=True, train=True, transform=trans
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_folder, download=True, train=False, transform=trans
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def train_model(train_data, num_epochs=5, learning_rate=1e-4,
                device=torch.device("cpu")):
    '''
    This will train the NN.
    '''
    model = ConvNet(device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    acc_list = []
    ppb_length = 50
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data):
            # Run the forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                n_eq = int(
                    (epoch * len(train_data) + i) * ppb_length /
                    (num_epochs * len(train_data))
                )
                n_dash = ppb_length - n_eq
                line = "\r    |" + "=" * n_eq + "-" * n_dash +\
                    "| loss = %.2f " % loss +\
                    "Train Accuracy = %.2f" % (acc_list[-1] * 100.0)
                line = line + " " * (99 - len(line))
                sys.stdout.write(line)
                sys.stdout.flush()

    line = "\r    |" + "=" * ppb_length +\
        "| loss = %.2f " % loss +\
        "Train Accuracy = %.2f" % (acc_list[-1] * 100.0)
    line = line + " " * (99 - len(line))
    sys.stdout.write(line)
    sys.stdout.flush()

    return model


def get_model_accuracy(model, test_data):
    '''
    '''
    # We must turn it to eval mode to be able to test the model.  This
    # prevents gradient calculations to be done.
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100.0 * correct / total


if __name__ == "__main__":
    # device = torch.device("cpu")
    device = torch.device("cuda:0")

    train = True
    train_loader, test_loader = setup()
    if train:
        t0 = time.time()
        model = train_model(train_loader, device=device)
        t1 = time.time()
        print("\nTime to train model = %.2f seconds." % (t1 - t0))
        torch.save(model.state_dict(), 'mnist_conv.nn')
    else:
        model.load_state_dict(torch.load("mnist_conv.nn"))
        model.eval()
    accuracy = get_model_accuracy(model, test_loader)
    print("Model accuracy to test set is %.2f%%" % accuracy)
