import tensorflow as tf
import flwr as fl


# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Load and prepare model
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class FlowerClient(fl.client.NumPyClient):
    # Get the current weights of the model
    def get_parameters(self, config):
        return model.get_weights()
    
    # Train the model with the received parameters (weights) from the server
    def fit(self, parameters, config):
        # Set the model weights
        model.set_weights(parameters) 
        # Train the model on local data
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        # Return the updated weights and number of training samples
        return model.get_weights(), len(x_train), {} 
    
    # Evaluate the model on local test data to calculate performance (e.g., accuracy)
    def evaluate(self, parameters, config):
        # Set the model weights
        model.set_weights(parameters)
        # Evaluate the model on test data
        loss, accuracy = model.evaluate(x_test, y_test)
        # Return the loss and accuracy metrics
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient()
)