# Federated Learning Implementation with Flower

This project demonstrates how federated learning can be implemented using Flower (FL). Flower is a framework that enables machine learning on decentralized data, making it suitable for privacy-preserving applications and edge-device learning. The setup consists of three components: a server, clients, and a centralized script that handles the model and data.

## Project Overview

In this example, we are using **Flower** to implement federated learning, where a neural network model is trained on local client devices using a dataset (CIFAR-10). The model is then aggregated by a central server to improve its generalization. The setup includes three files:

1. **server.py**: The central server that manages the federated learning process.
2. **client.py**: The client that holds the model, trains it on local data, and communicates with the server.
3. **centralized.py**: A script to define the neural network, load data, train, and evaluate the model.

The model used is a convolutional neural network (CNN) that is trained on the CIFAR-10 dataset, which is a collection of 60,000 32x32 color images in 10 classes.

## Files Explanation

### 1. **server.py**

This file defines the federated learning server that runs the aggregation of model updates from all clients. It uses **FedAvg** (Federated Averaging), which averages the model parameters received from clients after each round of training.

- **`weighted_average`**: This function calculates the weighted average of the accuracy metrics from clients during evaluation.
- **`fl.server.start_server`**: Starts the server on the specified IP and port and runs the federated learning process.

The server coordinates the federated learning process across clients. It waits for updates from the clients after each round and then averages the model weights to create a global model.

### 2. **client.py**

This file defines the federated learning client. Each client holds a local copy of the model, trains it on local data, and sends updates to the server.

- **`set_parameters`**: A function to update the local model with the global parameters received from the server.
- **`FlowerClient`**: The main client class that:
  - **`get_parameters`**: Returns the model parameters in NumPy format to send to the server.
  - **`fit`**: Trains the model on local data for one epoch using the `train` function.
  - **`evaluate`**: Evaluates the model on local test data and returns the evaluation results to the server.

This file also connects to the server, receives model parameters, and participates in the federated learning process.

### 3. **centralized.py**

This file defines the neural network model architecture (CNN for CIFAR-10), training and evaluation functions, and data loading functionality. This file is intended to be used locally on each client and also provides a script for centralized training (if needed).

- **`Net`**: The neural network model, a CNN with two convolutional layers followed by fully connected layers.
- **`train`**: The function to train the model using stochastic gradient descent (SGD) with Cross-Entropy loss.
- **`test`**: Evaluates the model on the test dataset, calculating both loss and accuracy.
- **`load_data`**: Loads and preprocesses the CIFAR-10 dataset, returning DataLoader objects for training and testing.
- **`load_model`**: Loads and instantiates the neural network model.

In the federated setting, this file is used by each client to load data, define the model, and perform local training and evaluation.

### How the Federated Learning Works

1. **Client Initialization**: Each client initializes a copy of the model and loads the CIFAR-10 dataset.
2. **Server Initialization**: The server initializes a federated learning process where it waits for updates from the clients.
3. **Training**:
   - Each client trains the model locally on its dataset for a specified number of epochs.
   - The updated model parameters are sent to the server.
4. **Model Aggregation**: The server collects model updates from clients and averages the parameters to create a global model.
5. **Evaluation**: The server can periodically evaluate the aggregated model’s performance.
