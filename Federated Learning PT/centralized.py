import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Define the device to be used for training (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network architecture using a class that inherits from nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: input channels = 3 (RGB), output = 6, kernel size = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer: 2x2 window, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: input channels = 6, output = 16, kernel size = 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # First fully connected layer: flattening 16 feature maps of 5x5 to 120 units
        self.fc1 = nn.Linear(16*5*5, 120)
        # Second fully connected layer: 120 units to 84
        self.fc2 = nn.Linear(120, 84)
        # Output layer: 84 units to 10 (one for each CIFAR-10 class)
        self.fc3 = nn.Linear(84, 10)
    
    # Define the forward pass (how data flows through the network)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
        x = x.view(-1, 16*5*5)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # FC1 → ReLU
        x = F.relu(self.fc2(x))  # FC2 → ReLU
        return self.fc3(x)       # FC3 (no activation since it's handled by loss function)

# Train the model for a specified number of epochs using SGD and cross-entropy loss
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()  # Clear gradients
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()  # Backpropagation
            optimizer.step()  # Update model parameters

# Evaluate the model on the test dataset, return loss and accuracy
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():  # No gradient needed for evaluation
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))  # Forward pass
            loss += criterion(outputs, labels.to(DEVICE)).item()  # Accumulate loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()  # Count correct predictions
    return loss / len(testloader.dataset), correct / total  # Return average loss and accuracy

# Load and preprocess CIFAR-10 dataset, return train and test data loaders
def load_data():
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# Instantiate the model and move it to the appropriate device
def load_model():
    return Net().to(DEVICE)

# Entry point: load model and data, train the model, test it, and print performance
if __name__ == "__main__":
    net = load_model()
    trainloader, testloader = load_data()
    train(net, trainloader, 5)
    loss, accuracy = test(net, testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
