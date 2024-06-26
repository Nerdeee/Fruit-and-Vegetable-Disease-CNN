import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F

X_list = []
y_list = []
class_names = []

with open("features.pickle", "rb") as f:
    X_list = pickle.load(f)

with open("labels.pickle", "rb") as f:
    y_list = pickle.load(f)

with open("classes.pickle", "rb") as f:
    class_names = pickle.load(f)
# normalize the image data
X_list = X_list/255.0

# permute data to be in pytorch format, aka: (batch_size, channels, height, width)
X_list = np.transpose(X_list, (0,3,1,2))

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.int64)

# split 80% of training data

eighty_percent = int(len(X) * 0.8)

X_train = X[:eighty_percent]
y_train = y[:eighty_percent]

X_test = X[eighty_percent:]
y_test = X[eighty_percent:]

print("\n")
print('X\'s shape: ', X_train.shape)
print('y\'s shape: ', y_train.shape)

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
num_epochs = 25

print(len(X_train))

for epoch in range(num_epochs):
    total_correct = 0
    accuracy = 0.0
    total_loss = 0.0
    for i in range(len(X_train)):
        # forward pass
        img = X_train[i].unsqueeze(0)
        label = y_train[i]

        outputs = model(img)
        loss = loss_fn(outputs, label.unsqueeze(0))

        # print('model outputs type: ', type(outputs))

        total_loss += loss.item()
        _, prediction = torch.max(outputs.data, 1)
        # print('prediction.item() output ', prediction.item())
        # print('label output ', label.item())
        
        if class_names[prediction.item()] == label.item(): total_correct += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(X_train)
    accuracy = total_correct / len(X_train)
    print(f'Epoch: {epoch + 1} / {num_epochs}\tAccuracy: {accuracy:.2f}%\tLoss: {loss.item():.4f}')

print('Model training complete!')

# testing
'''
def validate(model, x_val, y_val, loss_fn):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(x_val)):
            inputs = x_val[i].unsqueeze(0)
            targets = y_val[i].unsqueeze(0)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            val_loss+=loss.item()
            _, predicted = torch.max()
'''