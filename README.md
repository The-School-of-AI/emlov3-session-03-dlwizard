[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ybfMCDlj)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11278941&assignment_repo_type=AssignmentRepo)
# Docker Compose Assignment: MNIST Training, Evaluation, and Inference

In this assignment, you will create a Docker Compose configuration to perform training, evaluation, and inference on the MNIST dataset.

Requirements:

1. Create three services in the Docker Compose file: **`train`**, **`evaluate`**, and **`infer`**.
2. Use a shared volume called **`mnist`** for data exchange between services.
3. The **`train`** service should:
    - Look for a checkpoint file in the volume. If found, resume training from that checkpoint. Train for 1 epoch and save the final checkpoint. Once done, exit.
    - Share the model code by importing the model instead of copy-pasting it.
4. The **`evaluate`** service should:
    - Look for the final checkpoint file in the volume. Evaluate the model using the checkpoint and save the evaluation metrics. Once done, exit.
    - Share the model code by importing the model instead of copy-pasting it.
5. The **`infer`** service should:
    - Wait for the evaluation to be completed. If the accuracy is greater than 95%, run inference on any 5 random MNIST images and save the results (images with file name as predicted number) in the **`results`** folder in the volume. Then exit.
    - Share the model code between services by importing the model instead of copy-pasting it in each service file.
6. After running all the services, ensure that the data, model, and results are available in the **`mnist`** volume.

Detailed Instructions:

1. Build all the Docker images using **`docker-compose build`**.
2. Run the Docker Compose services using **`docker-compose run train`**, **`docker-compose run evaluate`**, and **`docker-compose run infer`**. Verify that all services have completed successfully.
3. Check if the checkpoint file (**`mnist_cnn.pt`**) is saved in the **`mnist`** volume. If found, display "Checkpoint file found." If not found, display "Checkpoint file not found!" and exit with an error.
4. Check if the evaluation results file (**`eval_results.json`**) is saved in the **`mnist`** volume.
5. Check the contents of the **`results`** folder in the **`mnist`** volume see if the inference results are saved.
6. Check the size of each Docker image and PyTorch version for the **`train`**, **`evaluate`**, and **`infer`** services. Ensure that the image size is not greater than 990MB and the PyTorch version is 1.10.0 or higher. Display appropriate messages for each service indicating whether it meets the size and version requirements.

The provided grading script will run the Docker Compose configuration, check for the required files, display the results, and perform size and version checks.