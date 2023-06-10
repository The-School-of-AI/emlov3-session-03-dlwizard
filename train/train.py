import os
from tqdm import tqdm
import click
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import sys

# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
parent = os.path.dirname(current)
 
# adding the parent directory to the sys.path.
sys.path.append(parent)

from model import Net

best_acc = 0
start_epoch = 0

def train(
    model: Net,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.SGD,
    scheduler: StepLR,
    criterion: nn.CrossEntropyLoss,
    l1_reg: float = None,
) -> Tuple[float, float]:
    """This function trains the model for one epoch.
    Parameters
    ----------
    model: Net
        The model to train

    device: torch.device
        The device to train the model on

    train_loader: DataLoader
        The data loader to use for training

    optimizer: optim.SGD
        The optimizer to use for training

    scheduler: StepLR
        The scheduler to use for training

    criterion: nn.CrossEntropyLoss
        The loss function to use for training

    l1_reg: float
        The L1 regularization factor to use for training



    Returns
    -------
    epoch_accuracy: float
        The accuracy of the model on the training set

    epoch_loss: float
        The loss of the model on the training set

    """

    model.train()

    # collect stats - for accuracy calculation
    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(train_loader)

    for batch_id, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Gather prediction and calculate loss + backward pass + optimize weights
        label_pred = model(data)
        label_loss = criterion(label_pred, target)

        # L1 regularization
        if l1_reg is not None:
            l1_criterion = nn.L1Loss(size_average=False)
            l1_reg_loss = 0
            for param in model.parameters():
                l1_reg_loss += l1_criterion(param, torch.zeros_like(param))
                # print("L1 reg loss: ", l1_reg_loss)
            label_loss += l1_reg * l1_reg_loss

        # Calculate gradients
        label_loss.backward()
        # Optimizer
        optimizer.step()

        # Metrics calculation- For epoch Accuracy(total correct pred/total items) and loss
        pred = label_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        epoch_loss += label_loss.item()
        pbar.set_description(
            desc=f"Training Set: Loss={epoch_loss/len(train_loader)}, Batch_id={batch_id}, Train Accuracy={100*correct/processed:0.2f}"
        )

    epoch_accuracy = 100 * correct / processed
    epoch_loss /= len(train_loader)
    # scheduler.step(epoch_loss/len(train_loader))

    scheduler.step(epoch_loss)

    return epoch_accuracy, epoch_loss


def test(
    model_dir: str,
    name: str,
    model: Net,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    epoch: int,
    best_acc: float,
    save_model: bool,
) -> Tuple[float, float, float]:
    """This function tests the model for one epoch.

    Parameters
    ----------
    name: str
        The name of the model

    model: Net
        The model to test

    device: torch.device
        The device to test the model on

    test_loader: DataLoader
        The data loader to use for testing

    criterion: nn.CrossEntropyLoss
        The loss function to use for testing

    epoch: int
        The current epoch number

    best_acc: float
        The best accuracy achieved so far

    save_model: bool
        Whether to save the model or not

    Returns
    -------
    epoch_accuracy: float
        The accuracy of the model on the test set

    epoch_loss: float
        The loss of the model on the test set

    best_acc: float
        The best accuracy achieved so far

    """

    model.eval()

    correct = 0
    processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    pbar = tqdm(test_loader)

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device)

            label_pred = model(data)
            label_loss = criterion(label_pred, target)

            # Metrics calculation
            pred = label_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            epoch_loss += label_loss.item()
            pbar.set_description(
                desc=f"Test Set: Loss={epoch_loss/len(test_loader)}, Batch_id={batch_id}, Test Accuracy={100*correct/processed:0.2f}"
            )

    epoch_accuracy = (100 * correct) / processed
    epoch_loss /= len(test_loader)

    # Save checkpoint.
    if epoch_accuracy > best_acc:
        print("\n*****Saving Model*****")
        state = {
            "net": model.state_dict(),
            "acc": epoch_accuracy,
            "epoch": epoch,
        }
        if save_model:
            torch.save(state, os.path.join(model_dir, name))
        best_acc = epoch_accuracy

    return epoch_accuracy, epoch_loss, best_acc


def dataloader(
    root_dir: str, 
    cuda: str, 
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """This function creates the train and test data loaders.
    parameters
    ----------
    cuda: str
        The device to use for training and testing

    batch_size: int
        The batch size to use for training and testing

    Returns
    -------
    train_loader: DataLoader
        The data loader to use for training

    test_loader: DataLoader
        The data loader to use for testing

    """
    # Train Phase transformations
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Test Phase transformations
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_dir = os.path.join(root_dir, "data")

    train = datasets.MNIST(
        data_dir, train=True, download=True, transform=train_transforms
    )
    test = datasets.MNIST(
        data_dir, train=False, download=True, transform=test_transforms
    )

    batch_size = batch_size or (128 if cuda else 64)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = (
        dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True)
        if cuda
        else dict(shuffle=True, batch_size=64)
    )

    # train dataloader
    train_loader = DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = DataLoader(test, **dataloader_args)

    return train_loader, test_loader


def train_model(
    root_dir: str,
    model_name: str,
    model: Net,
    resume: bool,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    gamma: float,
    cuda: bool,
    save_model: bool,
) -> None:
    """This function trains the model. It also saves the model if the accuracy improves.

    Parameters
    ----------

    root_dir: str
        The root directory of the project

    model_name: str
        The name of the model

    model: Net
        The model to train

    resume: bool
        Whether to resume training from a checkpoint or not

    device: torch.device
        The device to train the model on

    epochs: int
        The number of epochs to train the model for

    batch_size: int
        The batch size to use for training and testing

    learning_rate: float
        The learning rate to use for training

    momentum: float
        The momentum to use for training

    gamma: float
        The gamma to use for training

    cuda: bool
        Whether to use cuda or not

    save_model: bool
        Whether to save the model or not

    """

    global start_epoch, best_acc
    # model = None
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    train_loader, test_loader = dataloader(root_dir, cuda, batch_size)

    image_shape = np.array(next(iter(train_loader))[0].shape[1:])
    print(
        "\n\n****************************************************************************\n"
    )
    print("*****Model Details*****\n")
    print(f"Input Image Size: {image_shape}\n")
    print("*****Training Parameters*****\n")
    print(f"No Of Epochs: {epochs}\n")
    print(f"Batch size: {batch_size}\n")
    print(f"Initial Learning Rate: {learning_rate}")
    print(
        "\n****************************************************************************\n"
    )
    model_dir = os.path.join(root_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        if resume:
            # Load checkpoint.
            print("==> Resuming from checkpoint..")
            checkpoint = torch.load(os.path.join(model_dir, model_name))
            model.load_state_dict(checkpoint["net"])
            print("==> Model loaded from checkpoint..")
            best_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]

    # Optimization algorithm from torch.optim
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=6, gamma=gamma)
    # Loss condition
    criterion = nn.CrossEntropyLoss()
    print(
        "\n****************************************************************************\n"
    )
    print("*****Training Starts*****\n")

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        print(f"Training Epoch: {epoch}")
        train_acc_delta, train_loss_delta = train(
            model, 
            device, 
            train_loader, 
            optimizer, 
            scheduler, 
            criterion
        )
        test_acc_delta, test_loss_delta, best_acc = test(
            model_dir,
            model_name,
            model,
            device,
            test_loader,
            criterion,
            epoch,
            best_acc,
            save_model,
        )
        # print(f"Learning Rate: {scheduler._last_lr[0]},{optimizer.param_groups[0]['lr']}")
        train_accuracy.append(round(train_acc_delta, 2))
        train_loss.append(round(train_loss_delta, 4))
        test_accuracy.append(round(test_acc_delta, 2))
        test_loss.append(round(test_loss_delta, 4))

    print("*****Training Stops*****\n")


@click.command()
@click.option("--root_dir", default="./mnist", help="Root directory for data and model")
@click.option("--batch_size", default=128, help="Batch size for training")
@click.option("--epochs", default=1, help="Number of epochs to train")
@click.option("--lr", default=0.01, help="Learning rate for training")
@click.option("--dropout", default=0.1, help="Dropout value for training")
@click.option("--gamma", default=0.7, help="Learning rate decay factor")
@click.option("--momentum", default=0.9, help="SGD momentum")
@click.option("--seed", default=1, help="Seed value for training")
@click.option("--save_model", default=True, help="For Saving the current Model")
@click.option("--resume", default=True, help="Resume training from checkpoint")
def main(
    root_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    dropout: float,
    gamma: float,
    momentum: float,
    seed: int,
    save_model: bool,
    resume: bool,
) -> None:
    """This function is the main function that will be called from the command line.
    It will take all the arguments from the command line and pass it to the train functions.

    Parameters
    ----------

    root_dir: str
        The root directory for data and model

    batch_size: int
        The batch size to use for training and testing

    epochs: int
        The number of epochs to train the model for

    lr: float
        The learning rate to use for training

    dropout: float
        The dropout to use for training

    gamma: float
        The gamma to use for training

    momentum: float
        The momentum to use for training

    seed: int
        The seed to use for training

    save_model: bool
        Whether to save the model or not

    resume: bool
        Whether to resume training from a checkpoint or not

    """

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model: Net = Net(dropout).to(device)

    # root_dir = "./mnist"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    train_model(
        root_dir,
        "mnist_cnn.pt",
        model,
        resume,
        device,
        epochs,
        batch_size,
        lr,
        momentum,
        gamma,
        use_cuda,
        save_model,
    )


if __name__ == "__main__":
    # execute only if run as a script (not by 'import')
    main()
