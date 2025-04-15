## Introduction to Federated Learning (FL)

**Federated Learning (FL)** is a machine learning technique where multiple devices or nodes (clients) collaboratively train a model **without sharing their raw data**. Instead of sending data to a central server, each device trains the model locally and only shares **model updates (weights/gradients)**.

> Think of FL like a study group: each student studies a chapter on their own and only shares their notes (learnings), not the whole textbook (raw data).

This approach is great for:
- **Data privacy** (no raw data leaves the device)
- **Edge computing** (training on mobile devices, browsers, etc.)
- **Decentralized AI**

---

## Project Structure Overview

This example uses the **[Flower framework](https://flower.dev)**, which is designed to make federated learning easy with libraries like TensorFlow and PyTorch.

You have two main scripts:
1. `server.py` – acts as the **central coordinator**
2. `client.py` – acts as the **participant or worker**


## server.py — The Federated Learning Server

```python
import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3)
)
```

### What this does:
- **Initializes a server** on IP `0.0.0.0` and port `8080`
- **Coordinates 3 rounds of training**
- **Waits for clients to connect**, receive initial model weights, and collect updated ones after training

 The server **does not contain a model** — it only manages:
- Distribution of model weights
- Collection of updates from clients
- Aggregation of updates to improve the global model

## client.py — The Federated Learning Client

```python
import tensorflow as tf
import flwr as fl

# Load and prepare model
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

### What this does:
- Loads a **MobileNetV2 model** suitable for image classification
- Loads **CIFAR-10**, a dataset of 60,000 32x32 color images in 10 classes
- Compiles the model using:
  - **Adam** optimizer
  - **Sparse categorical crossentropy** loss
  - **Accuracy** metric

---

### The Flower Client Class

```python
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
```

#### Method Breakdown:
- `get_parameters()`: Sends current model weights to the server
- `fit()`: 
  - Receives global weights from the server
  - Trains locally on client data for 1 epoch
  - Returns updated weights and number of samples used
- `evaluate()`:
  - Sets weights from server
  - Evaluates model on local test data
  - Returns loss and accuracy metrics

### Start the Client

```python
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient()
)
```

- Connects to the server running at `127.0.0.1:8080`
- Registers this device as a **federated client**
- Starts the federated training process

---

## What Happens in a Full Federated Round?

1. **Server broadcasts weights** to clients
2. **Clients update weights** using local data
3. **Clients return weights** to the server
4. **Server aggregates** all updates
5. **Server sends improved model** back to clients
6. Repeat for 3 rounds (as defined)

## Summary

| Role        | What it does                            |
|-------------|------------------------------------------|
| **Server**  | Coordinates training, aggregates updates |
| **Client**  | Trains model locally, shares updates     |
| **Model**   | Defined locally on client                |
| **Data**    | Stays on the client — never shared       |


## **Final Emphasis:** Things to Note:

- Each client must define the **same model architecture**
- The server doesn’t know the model architecture, only weights
- Raw data **never leaves the client** — only model updates are shared
- The system is **privacy-preserving by design**