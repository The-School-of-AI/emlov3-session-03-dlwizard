import os
import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


def test_epoch(model, device, data_loader):
    # Test the model on the test dataset and calculate metrics
    # ...

    # Print the evaluation results
    # print(out)
    # ...

    # Return the evaluation results
    # return out
    pass


def main():
    # Initialize arguments
    # ...

    # Set device (CPU or GPU)
    # ...

    # Set data loaders
    # ...

    # Initialize the model
    # ...

    # Load the saved model checkpoint
    # saved_ckpt = Path(args.save_dir) / "model" / "mnist_cnn.pt"
    # model.load_state_dict(torch.load(saved_ckpt))
    # ...

    # Evaluate the model on the test dataset
    # eval_results = test_epoch(model, device, test_loader)
    # ...

    # Save the evaluation results to a JSON file
    # with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
    #     json.dump(eval_results, f)
    pass


if __name__ == "__main__":
    main()
