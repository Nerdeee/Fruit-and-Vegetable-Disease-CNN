import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F

X_list = []
y_list = []

with open("features.pickle", "rb") as f:
    X_list = pickle.load(f)

with open("labels.pickle", "rb") as f:
    y_list = pickle.load(f)

# normalize the image data
X_list = X_list/255.0

# permute data to be in pytorch format, aka: (batch_size, channels, height, width)
X_list = np.transpose(X_list, (0,3,1,2))

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.int64)


print("\n")
print('X\'s shape: ', X.shape)
print('y\'s shape: ', y.shape)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 28)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 35344)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = NeuralNet()

loss_fn = nn.CrossEntropyLoss()    
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

# training
num_epochs = 10

print(len(X))

for epoch in range(num_epochs):
    for i in range(len(X)):
        # forward pass
        img = X[i].unsqueeze(0)
        label = y[i]

        outputs = model(img)
        loss = loss_fn(outputs, label.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 800 == 0:
            print(f'Epoch: {epoch + 1} / {num_epochs}\tLoss: {loss.item():.4f}')

print('Model training complete!')

# testing
# with torch.no_grad():