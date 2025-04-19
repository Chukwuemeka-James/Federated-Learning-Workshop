# Federated Learning Workshop

Welcome to the **Federated Learning Workshop** hosted by the **CoLab ML Engineering Community**! This hands-on session introduces you to the core concepts of **federated learning** using the powerful [Flower](https://flower.dev/) framework with implementations in both **PyTorch** and **TensorFlow**.

## What Is Federated Learning?

**Federated Learning (FL)** is a machine learning technique that trains models across multiple decentralized devices or servers holding local data samples — **without exchanging their data**. This means:
- Data stays private and local
- Only model updates (gradients) are shared with the central server
- It's ideal for privacy-sensitive applications (e.g., healthcare, finance, smartphones)

## What is Flower?

**[Flower (FLWR)](https://flower.dev/)** is a flexible, open-source framework for building federated learning systems with popular ML libraries like PyTorch, TensorFlow, JAX, and more.

With Flower, you can:
- Simulate clients locally
- Customize training strategies
- Deploy across real-world edge devices or servers

---

## Two Implementations: PyTorch & TensorFlow

This workshop includes two fully functional FL implementations:

### Directory Structure

```
Federated Learning Workshop/
├── Federated Learning PT/
│   ├── client.py
│   ├── server.py
│   ├── centralised.py
│   ├── requirements.txt
|
├── Federated Learning TF/
│   ├── client.py
│   ├── server.py
│   ├── requirements.txt
```

---

## Setup Instructions

- We'll set up both environments separately using **virtual environments (venv)** for isolation and clarity.
- To implement this project, you need to have **`pipenv`** installed, you can do that using:
   ```bash
   pip install pipenv
   ```
- You'll also need to install **`python_version = 3.12`**

---

### PyTorch Version Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chukwuemeka-James/Federated-Learning-Workshop.git
   ```

2. **Navigate to the PyTorch project folder:**
   ```bash
   cd "Federated Learning Workshop/Federated Learning PT"
   ```

3.  **Creat and activate environment:**
   ```bash
   pipenv shell
   ```

4. **Install dependencies:**
   ```bash
   pipenv install
   ```

5. **Start the server (in a terminal):**
   ```bash
   python server.py
   ```

6. **Start at least two clients (each in a separate terminal):**
   ```bash
   python client.py
   ```

---

### TensorFlow Version Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/Chukwuemeka-James/Federated-Learning-Workshop.git
   ```

2. **Navigate to the TensorFlow project folder:**
   ```bash
   cd "Federated Learning Workshop/Federated Learning TF"
   ```

3.  **Creat and activate environment:**
   ```bash
   pipenv shell
   ```

4. **Install dependencies:**
   ```bash
   pipenv install
   ```

5. **Start the server (in a terminal):**
   ```bash
   python server.py
   ```

6. **Start at least two clients (each in a separate terminal):**
   ```bash
   python client.py
   ```

---

### **About CIFAR-10 Dataset**

In this workshop, we will be using the **CIFAR-10 dataset**, which is a widely used benchmark dataset in machine learning. It consists of 60,000 32x32 color images across 10 different classes. Each class has 6,000 images, and these classes are:

1. **Airplane**
2. **Automobile**
3. **Bird**
4. **Cat**
5. **Deer**
6. **Dog**
7. **Frog**
8. **Horse**
9. **Ship**
10. **Truck**

This dataset is used to train image classification models, and each image is labeled with one of these 10 classes. The model will learn to recognize and classify these different objects by training on the CIFAR-10 dataset.

---

### 📚 Acknowledgements

This workshop was inspired and guided by the following amazing resources:

- [Federated Learning with Flower and PyTorch](https://youtu.be/jOmmuzMIQ4c?si=xe1pY56TlvXKlqP8)  
- [Federated Learning with Flower and TensorFlow](https://youtu.be/FGTc2TQq7VM?si=aypC-94fuZ8hxtpX)

Special thanks to the creators of these tutorials for their clear and practical insights into federated learning with Flower.
