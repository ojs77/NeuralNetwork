# MNIST: handwritten digit pictures
# DataLoader, Transformation
# Multilayer NN, Activation Functions
# Loss and Optimizer
# Training Loop (Batches)
# Model Eval
# GPU Support


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784 # 28x28 size pictures
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# MNIST
train_data = torchvision.datasets.MNIST(root="./data", train = True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root="./data", train = False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_data, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = examples.next()

print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()

# Define NN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optim
# CEL does softmax at end so dont need to put in forward pass
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch: {epoch+1} / {num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}")

print("Finished Training")

# Model Eval
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    acc = 100 * n_correct/n_samples
    print(f"Acc: {acc:.3f}%")