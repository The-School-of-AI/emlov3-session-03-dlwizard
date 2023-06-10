import os
import json
import random
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def infer(
    root_dir: str,
    model_name: str,
    model: Net,
    no_of_images: int,
    device: torch.device,
) -> None:
    """This function is used to infer on the test data. It loads the model from the checkpoint and infers on the test data. 
    It saves the results in the results directory.

    Parameters
    ----------
    root_dir: str
        The root directory of the project

    model_name: str
        The name of the model

    model: Net
        The model to train

    no_of_images: int
        The number of images to infer on

    device: torch.device
        The device to train the model on

    """

    global start_epoch, best_acc

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_dir = os.path.join(root_dir, "data")

    test = datasets.MNIST(
        data_dir, train=False, download=True, transform=test_transforms
    )

    model_dir = os.path.join(root_dir, "model")

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    checkpoint = torch.load(os.path.join(model_dir, model_name))
    model.load_state_dict(checkpoint["net"])
    print("==> Model loaded from checkpoint..")

    print(
        "\n****************************************************************************\n"
    )
    print("*****Inference Starts*****\n")
    results_dir = os.path.join(root_dir, "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    model.eval()
    from torchvision.utils import save_image

    with torch.no_grad():
        sample_index = random.sample(range(len(test)), no_of_images)
        for i in sample_index:
            data = test[i][0].unsqueeze(0).to(device)
            target = torch.tensor(test[i][1]).to(device)
            label_pred = model(data)
            pred = label_pred.argmax(dim=1, keepdim=True)[0][0]
            filename = (
                f"predicted_{pred.item()}_label_{target.item()}_index_{str(i)}.png"
            )
            save_image(data, os.path.join(results_dir, filename))

    print("*****Inference Stops*****\n")


@click.command()
@click.option("--dropout", default=0.1, help="Dropout value")
@click.option("--n", default=5, help="Number of images to infer on")
@click.option("--seed", default=1, help="Seed value for training")
def main(
    dropout: float,
    n: int,
    seed: int,
) -> None:
    """This is the main function which is used to infer on the test data. 
    It loads the model from the checkpoint and infers on the test data.

    Parameters
    ----------

    dropout: float
        The dropout to use for training

    no_of_images: int
        The number of images to infer on

    seed: int
        The seed to use for training

    """

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model: Net = Net(dropout).to(device)

    root_dir = "./mnist"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    model_dir = os.path.join(root_dir, "model")
    data = json.load(open(os.path.join(model_dir, "eval_results.json")))
    if data["Accuracy"] > 95:
        infer(
            root_dir,
            "mnist_cnn.pt",
            model,
            n,
            device,
        )
    else:
        print("Accuracy is less than 95. Not inferring on test data.")


if __name__ == "__main__":
    # execute only if run as a script (not by 'import')
    main()
