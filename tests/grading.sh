#!/bin/bash

set -e # Exit immediately if any command fails

export COMPOSE_PROJECT_NAME=mnist

echo "🏗 Building all images"
docker-compose build

# Step 1: Run the docker-compose services
echo "🚀 Running the docker-compose services..."
docker-compose run train
docker-compose run evaluate
docker-compose run infer
echo "✅ All services have completed."

# Step 2: Check if the checkpoint is saved in the volume
echo "🔍 Checking for checkpoint file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_mnist:/opt/mount busybox ls /opt/mount/model/mnist_cnn.pt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Checkpoint file found."
else
    echo "❌ Checkpoint file not found!"
    exit 1
fi

# Step 3: Check if the eval.json output is saved in the volume
echo "🔍 Checking for eval_results.json file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_mnist:/opt/mount busybox ls /opt/mount/model/eval_results.json > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ eval_results.json file found."
else
    echo "❌ eval_results.json file not found!"
    exit 1
fi

# Step 4: Print the output of eval_results.json
echo "📄 Printing the content of eval_results.json file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_mnist:/opt/mount busybox cat /opt/mount/model/eval_results.json

# Step 5: Print the contents of the results folder in the volume
echo "📂 Printing the contents of the results folder..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_mnist:/opt/mount busybox ls /opt/mount/results

# Step 6: Check the size of each Docker image and PyTorch version
echo "📏 Checking the size of each Docker image and PyTorch version..."
for service in train evaluate infer; do
    docker-compose build ${service}
    image_id=$(docker images -q ${COMPOSE_PROJECT_NAME}_${service})
    image_size=$(docker inspect ${image_id} --format='{{.Size}}')
    pytorch_version=$(docker run --rm ${image_id} python -c "import pkg_resources; print(pkg_resources.get_distribution('torch').version.split('+')[0])")

    pytorch_version_check=$(python3 -c "import pkg_resources; print(pkg_resources.parse_version('${pytorch_version}') >= pkg_resources.parse_version('1.10.0'))")
    
    if [ $(($image_size/1000000)) -le 990 ] && [ ${pytorch_version_check} = "True" ]; then
        echo "✅ ${service} image is of valid size and PyTorch version is ${pytorch_version}."
    else
        echo "❌ ${service} image exceeds the valid size (990MB, actual: $(($image_size/1000000))MB) or PyTorch version ${pytorch_version} is less than 1.10!"
        exit 1
    fi
done
