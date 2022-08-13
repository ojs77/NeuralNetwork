import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Added additional conv layer and increased output channels per layer
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Input for fc1 matches x.view rescale in forward pass
        self.fc1 = nn.Linear(128 * 2**2, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output size 10 as 10 different classes
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace = False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128*2**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # No activation function at the end

        return x

# Class commented out as evaluating improved model currently, to evaluate original model, uncomment this class and comment out the above class.
# class ConvNet(nn.Module):
#     def __init__(self) -> None:
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16*5**2, 120)
#         self.fc2 = nn.Linear(120, 84)
#         # Output size 10 as 10 different classes
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5**2)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         # No activation function at the end

#         return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 4

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download= True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

model = torch.load("model_improved.pth").to( device)
model.eval()

# Model Eval, inside with loop as gradients do not need to be tracked - save time and space.
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    acc = [0 for i in range(10)]
    ppv = [0 for i in range(10)]
    f1_score = [0 for i in range(10)]
    n_class_correct = [0 for i in range(10)]
    n_class_false = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            else:
                n_class_false[pred] += 1
            n_class_samples[label] += 1

    # Retrieving statistics for each class
    for i in range(10):
        acc[i] = n_class_correct[i] / n_class_samples[i]
        ppv[i] = n_class_correct[i] / (n_class_correct[i] + n_class_false[i])
        f1_score[i] = 2 * acc[i] * ppv[i] /(acc[i] + ppv[i])
        print(f'Accuracy of {classes[i]}: {100.0 * acc[i]:.1f} %')
        print(f'PPV of {classes[i]}: {100.0 * ppv[i]:.1f} %')
        print(f"F1 Score of {classes[i]}: {100.0 * f1_score[i]:.1f} %")

    acc_overall = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc_overall} %')
    
    ppv_overall = 0
    for i in range(len(ppv)):
        ppv_overall += ppv[i]*n_class_samples[i]
    ppv_overall /= n_samples

    print(f"PPV of the network: {(100.0 *ppv_overall):.1f} %")
    
    f1_score_overall = 0
    for i in range(len(f1_score)):
        f1_score_overall += f1_score[i]*n_class_samples[i]
    f1_score_overall /= n_samples

    print(f"F1-Score of the network: {(100.0 * f1_score_overall):.1f} %")
    

    
    
