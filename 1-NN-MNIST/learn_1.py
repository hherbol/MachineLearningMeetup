'''
Learning exercise where we find weights to learn the MNIST handwritten
digit data set.
'''
# import os
import sys
import gzip
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt


def plot_image(d):
    '''
    Given the numpy or torch data set for an image, plot it.

    **Parameters**

        d: *torch.Tensor* or *np.array*
            Image data.

    **Returns**

        None
    '''
    image = np.asarray(d).squeeze()
    width = int(len(image)**(0.5))
    image = image.reshape(width, -1)
    plt.imshow(image)
    plt.show()


def parse_imgs(fname, num_images=None, N_SHOW=None):
    '''
    Parse the input of the image data set to torch tensors.

    **Parameters**

        fname: *str*
            File name.
        num_images: *int, optional*
            How many images to read in.  If None, then read in everything.
        N_SHOW: *int, optional*
            How many to display.

    **Returns**

        x: *torch.Tensor*
            Torch tensor of file data.

    **References**

        - https://stackoverflow.com/a/53570674
    '''
    sys.stdout.write("Parsing images from MNIST data set...")
    sys.stdout.flush()
    # Define some constants
    IMAGE_SIZE = 28
    IMAGE_SIZE_SQR = int(IMAGE_SIZE ** 2)

    fptr = gzip.open(fname, 'r')

    # Ignore header
    fptr.read(16)

    # If we only want to read N images, do so
    if num_images is None:
        # We know the data is stored as 28x28, so figure out num_images
        buf = fptr.read()
        num_images = int(len(buf) / IMAGE_SIZE_SQR)
    else:
        buf = fptr.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    # print(len(buf))
    # Parse from uint8 into float, as we will do math on it later
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)

    if N_SHOW is not None:
        for i, d in enumerate(data):
            if i > N_SHOW:
                break
            plot_image(d)

    xdata = torch.from_numpy(data.reshape(-1, IMAGE_SIZE_SQR))
    # We normalize it to make it better suited for a NN
    # In this case, I'm lazy and just divide by 255; however, a better
    # approach may be to normalize so each vector has mean 0 and std 1.
    sys.stdout.write(" DONE!\n")
    sys.stdout.flush()
    return xdata / 255.0


def parse_labels(fname, num_labels=None, N_SHOW=None):
    '''
    Parse the input of the label data set to torch tensors.

    **Parameters**

        fname: *str*
            File name.
        num_labels: *int, optional*
            How many labels to read in.  If None, then read in everything.

    **Returns**

        x: *torch.Tensor*
            Torch tensor of file data.
    '''
    sys.stdout.write("Parsing labels from MNIST data set...")
    sys.stdout.flush()

    fptr = gzip.open(fname, 'r')

    # Ignore header?
    fptr.read(8)

    # If we only want to read N images, do so
    if num_labels is None:
        # Each label should be an integer.
        buf = fptr.read()
        num_labels = int(len(buf))
    else:
        buf = fptr.read(num_labels)

    # Parse from uint8 into float, as we will do math on it later
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = torch.from_numpy(data.reshape(num_labels, 1))

    # THERE HAS TO BE A BETTER WAY IN TORCH TO DO THIS
    data2 = torch.zeros(len(data), 10)
    for i, d in enumerate(data):
        data2[i][int(d)] = 1

    sys.stdout.write(" DONE!\n")
    sys.stdout.flush()

    return data2


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.

    **References**

        - https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be
        used to stash information for backward computation. You can cache
        arbitrary objects for use in the backward pass using the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of
        the loss with respect to the output, and we need to compute the
        gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def learn(x, y, tx, ty, batch_size=64, hidden_dim=32, learning_rate=1e-4):
    '''
    Given training data, we try to find the weights to some NN model that we
    can use to predict what digit an image is.

    **Parameters**

        train_x: *torch.Tensor*
            4D Tensor data set of N images, each 28x28, of a single value that
            hold the intensity of an image pixel.  Each image is a digit from
            the MNIST data set.

        train_y: *torch.Tensor*
            2D Tensor data set of the corresponding actual digit associated
            with each image in train_x.

    **Returns**

        weights: *torch.Tensor*
            ?????

    **References**

        - https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''

    sys.stdout.write("Training NN\n")
    sys.stdout.flush()

    # Input and output dimensions of the NN
    D_in = x.size()[1]
    D_out = y.size()[1]

    # device = torch.device("cpu")
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    x = x.to(device)
    y = y.to(device)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, D_out),
    )
    model.cuda(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the
    # weights of the model for us. Here we will use Adam; the optim
    # package contains many other optimization algoriths. The first argument
    # to the Adam constructor tells the optimizer which Tensors it should
    # update.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = float('inf')
    MAX_ITER = 10000  # 00
    LOSS_GOAL = 1E-4
    iteration = 0
    total_update = 50
    iteration_update = int(MAX_ITER / total_update)
    while iteration < MAX_ITER and loss > LOSS_GOAL:
        if not iteration % iteration_update:
            n_eq = int(iteration / iteration_update)
            n_dash = total_update - n_eq
            line = "\r    |" + "=" * n_eq + "-" * n_dash +\
                "| loss = %.2f " % loss +\
                "RMSE = %.2f" % RMSE(tx, ty, model)
            line = line + " " * (79 - len(line))
            sys.stdout.write(line)
            sys.stdout.flush()
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        # if iteration % 100 == 0:
        #     print(iteration, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        iteration += 1

    torch.cuda.synchronize()
    line = "\r    |" + "=" * total_update + "| loss = %.2f" % loss
    line = line + " " * (79 - len(line))
    sys.stdout.write(line + "\nDONE!\n")

    # Test out saving the model to a file
    sys.stdout.write("Saving NN model to mnist.nn...")
    sys.stdout.flush()
    torch.save(model.state_dict(), "mnist.nn")
    sys.stdout.write(" DONE!\n")
    sys.stdout.flush()

    return model


def RMSE(test_x, test_y, model, verbose=False):
    '''
    Given data, test it and see what the RMS error is.
    '''

    # with model.no_grad():
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    err = []
    if verbose:
        print("Test    Model")
    for x, y in zip(test_x, test_y):
        model.eval()
        yhat = model(x)
        i1 = np.nanargmax(torch.Tensor.cpu(y))
        i2 = np.nanargmax(torch.Tensor.cpu(yhat).detach())
        err.append((i1 - i2)**2)
        if verbose:
            print("%d       %d" % (i1, i2))
    return np.square(np.mean(np.sqrt(np.array(err))))


def load_saved_model():
    # Test out re-loading everything
    sys.stdout.write("Loading NN model from mnist.nn...")
    sys.stdout.flush()
    D_in, D_out, hidden_dim = 784, 10, 32
    # device = torch.device("cpu")
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, D_out),
    )
    model.cuda(device)
    model.load_state_dict(torch.load("mnist.nn"))
    model.eval()
    sys.stdout.write(" DONE!\n")
    sys.stdout.flush()
    return model


if __name__ == "__main__":
    print("Task 1 - MNIST NN Test Case")
    print("-" * 79)

    train_model = False

    if train_model:
        train_x = parse_imgs("train/train-images-idx3-ubyte.gz")
        test_x = parse_imgs("test/t10k-images-idx3-ubyte.gz")
        train_y = parse_labels("train/train-labels-idx1-ubyte.gz")
        test_y = parse_labels("test/t10k-labels-idx1-ubyte.gz")

        # Train our model
        model = learn(train_x, train_y, test_x, test_y)

        # Test it out quickly
        model_rmse = RMSE(test_x, test_y, model)
        print("Final RMSE Error = %.2f" % model_rmse)
    else:
        test_x = parse_imgs("test/t10k-images-idx3-ubyte.gz")
        test_y = parse_labels("test/t10k-labels-idx1-ubyte.gz")

        model = load_saved_model()
        # Test out RMSE again to see if we loaded it correctly
        model_rmse = RMSE(test_x, test_y, model)
        print("Final RMSE Error = %.2f" % model_rmse)
