# Prepare data
# Design Model (input, output size, forward pass)
# Construct Loss and optimizer
# Training Loop
#   Forwrad Pass
#   Backward Pass
#   Update Weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model
# f = wx + b, sigmoid at end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    

model = LogisticRegression(n_features)

# Loss and optimizer
learning_rate = 0.01
criteriion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Training loop

num_epochs = 100

for epoch in range(num_epochs):
    # Forward and loss
    y_pred = model(X_train)
    loss = criteriion(y_pred, y_train)

    # Backward Pass
    loss.backward()

    # Update Weights
    optimizer.step()
    optimizer.zero_grad()

    # Statement
    if (epoch +1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy = {acc:.4f}")