import os
import json
from tqdm import tqdm
import click
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

def test(
    model: Net,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> Tuple[float, float, float]:
    """This function tests the model.

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
    out: dict
        The evaluation metrics

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

    out = {"Test loss": epoch_loss, "Accuracy": epoch_accuracy}
    return out


def dataloader(
    root_dir: str, 
    cuda: str, 
    batch_size: int = None
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


def eval_model(
    root_dir: str,
    model_name: str,
    model: Net,
    batch_size: int,
    device: torch.device,
    cuda: bool,
) -> dict:
    """This function evaluates the model. It loads the model from the checkpoint
    and then evaluates it. It returns the evaluation metrics.

    Parameters
    ----------
    root_dir: str
        The root directory of the project

    model_name: str
        The name of the model

    model: Net
        The model to train

    batch_size: int
        The batch size to use for training and testing

    device: torch.device
        The device to train the model on

    cuda: bool
        Whether to use cuda or not

    Returns
    -------
    eval_results: dict
        The evaluation metrics

    """

    global start_epoch, best_acc

    _, test_loader = dataloader(root_dir, cuda, batch_size)

    model_dir = os.path.join(root_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    checkpoint = torch.load(os.path.join(model_dir, model_name))
    model.load_state_dict(checkpoint["net"])
    print("==> Model loaded from checkpoint..")

    # Loss condition
    criterion = nn.CrossEntropyLoss()
    print(
        "\n****************************************************************************\n"
    )
    print("*****Evaluation Starts*****\n")

    eval_results = test(
        model,
        device,
        test_loader,
        criterion,
    )

    with open(os.path.join(model_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)

    print("*****Evaluation Stops*****\n")


@click.command()
@click.option("--root_dir", default="./mnist", help="Root directory of the project")
@click.option("--dropout", default=0.1, help="Dropout value")
@click.option("--batch_size", default=128, help="Batch size for testing")
@click.option("--seed", default=1, help="Seed value for training")
def main(
    root_dir: str,
    dropout: float,
    batch_size: int,
    seed: int,
) -> None:
    """This function is the main function that evaluates the model. It loads the model from the checkpoint and evaluates it.

    Parameters
    ----------

    root_dir: str
        The root directory of the project

    dropout: float
        The dropout to use for training

    batch_size: int
        The batch size to use for testing

    seed: int
        The seed to use for training

    """

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model: Net = Net(dropout).to(device)

    # root_dir = "./mnist"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    eval_model(
        root_dir,
        "mnist_cnn.pt",
        model,
        batch_size,
        device,
        use_cuda,
    )


if __name__ == "__main__":
    # execute only if run as a script (not by 'import')
    main()
