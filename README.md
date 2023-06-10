[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ybfMCDlj)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11278941&assignment_repo_type=AssignmentRepo)
# emlov3-session-03
# MNIST Training, Evaluation, and Inference using Docker Compose

## Introduction
Here we have used docker compose to train, evaluate and inference the MNIST dataset. We have used a shared volume named `mnist` to share the data between the containers. The data is downloaded from the internet and saved in the `data` directory inside it.
We have used PyTorch to train the model. The model is trained on the MNIST dataset and the trained model is saved in the `models` directory. The trained model is then used to evaluate the model on the test dataset and the accuracy matric is saved in `eval_results.json` file. The trained model is also used to inference the model on the test dataset and the inference results are saved in the `results` directory in mnist volume. Model code is also shared between the containers. Steps to run the training, evaluation and inference are given below:

## Install Docker
Follow the instructions on the [Docker website](https://docs.docker.com/install/) to install Docker on your machine.

## Install Docker Compose
Follow the instructions on the [Docker website](https://docs.docker.com/compose/install/) to install Docker Compose on your machine.

## Clone the Repository
Clone the repository to your local machine using the following command:

```
git clone repo link
```

## Build the Docker Images
Build the Docker images using the following command:

```
docker-compose build
```

## Run the training Docker Container
To run the training Docker container, use the following command:

```
docker-compose run train
```

## Run the evaluation Docker Container
To run the evaluation Docker container, use the following command:

```
docker-compose run evaluate
```

## Run the inference Docker Container
To run the inference Docker container, use the following command:

```
docker-compose run infer
```

# References
- [PyTorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [click](https://click.palletsprojects.com/en/7.x/)


# Team Members
- [Santu Hazra](https://www.linkedin.com/in/santuh)
- [Aditya Sharma](https://www.linkedin.com/in/iamditya)
- [Rakesh Swain](https://www.linkedin.com/in/rakesh-swain-59140075)
- [Amitabh Gupta](https://www.linkedin.com/mwlite/in/amitabh-gupta-13839b35)
- [Rakesh Raushan](https://www.linkedin.com/in/rakesh91)
