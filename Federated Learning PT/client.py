from collections import OrderedDict
import flwr as fl
import torch
from centralized import load_data, load_model, train, test

# Set model parameters using the parameters received from the server
def set_parameters(model, parameters):
    # Pair each parameter key in the model's state dict with its corresponding received value
    params_dict = zip(model.state_dict().keys(), parameters)
    # Create an ordered dictionary by converting each parameter value to a PyTorch tensor
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})    
    model.load_state_dict(state_dict, strict=True)


net = load_model()
trainloader, testloader = load_data()

# Define the federated learning client class using Flower's NumPyClient interface
class FlowerClient(fl.client.NumPyClient):

    # Method to get the current model parameters
    # These parameters will be sent to the federated server during initialization or updates
    def get_parameters(self, config):
        # Convert model state dict values to NumPy arrays and return them
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    # Method to train (fit) the model on local data
    # It receives the global model parameters from the server
    def fit(self, parameters, config):
        # Load the global model parameters into the local model
        set_parameters(net, parameters)
        # Train the model on the local training dataset for 1 epoch
        train(net, trainloader, epochs=1)
        # Return the updated parameters, the number of training examples, and an empty dictionary
        return self.get_parameters({}), len(trainloader.dataset), {}
    
    # Method to evaluate the model performance on local test data
    def evaluate(self, parameters, config):
        # Load the global model parameters into the local model
        set_parameters(net, parameters)
        # Test the model on the local test dataset and return the loss and accuracy
        loss, accuracy = test(net, testloader)
        # Return the evaluation loss, the number of test examples, and accuracy in a metrics dictionary
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)