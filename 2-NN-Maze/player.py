'''
Main background for this found here:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
'''
import copy
import time
import torch
from utils_imgs import *
from game import play_maze


class ConvNet(torch.nn.Module):
    '''
    This is our custom model that we are building.  For now, it is copied
    from the tutorial.
    '''

    def __init__(self, device=torch.device("cpu"), hidden_dim=64):
        super(ConvNet, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 7)
        )

        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = torch.nn.Dropout()
        # self.fc1 = torch.nn.Linear(7 * 7 * 64, 1000)
        # self.fc2 = torch.nn.Linear(1000, 10)
        self.device = device
        if "cuda" in str(device):
            self.cuda(device)

    def forward(self, x):
        '''
        This outlines how the data flows through this NN.
        '''
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        return out


def train_model(maze, nBlocks,
                max_iter=1000, max_game_loop=1000,
                learning_rate=1e-4,
                device=torch.device("cpu")):
    '''
    This will train the NN.
    '''
    model = ConvNet(device=device)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(max_iter):
        # Run the forward pass
        count, _ = play_maze(model, copy.deepcopy(maze),
                             nBlocks, max_game_loop=max_game_loop)
        loss = loss_fn(torch.Tensor([count]), torch.Tensor([0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(loss)

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
    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    net_name = "mazeRunner.nn"
    maze, nBlocks = load_maze("maze.png")

    train = True
    if train:
        t0 = time.time()
        model = train_model(
            copy.deepcopy(maze), nBlocks,
            max_iter=100, max_game_loop=100,
            device=device
        )
        t1 = time.time()
        print("\nTime to train model = %.2f seconds." % (t1 - t0))
        torch.save(model.state_dict(), net_name)
    else:
        model = ConvNet(device=device)
        model.load_state_dict(torch.load(net_name))
        model.eval()

    lowest_iter, maze_solve = play_maze(model, copy.deepcopy(maze), nBlocks)

    save_maze(maze_solve, name="solution")
