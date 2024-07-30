import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

fully_connected_layers = 4
learning_rate = 0.0007
dropout_probability = 0.1
experiment_name = f'LearningRate_{learning_rate} DropoutProbability_{dropout_probability} FCL_{fully_connected_layers}'
experiment_dir_name = os.path.join('runs', experiment_name)
os.mkdir(experiment_dir_name)

writer = SummaryWriter(log_dir=f'runs/{experiment_name}')

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

print(X[0])
print('X\'s shape: ', X.shape)
print('y\'s shape: ', y.shape)

# split 80% of training data

eighty_percent = int(len(X) * 0.8)

X_train = X[:eighty_percent]
y_train = y[:eighty_percent]

X_test = X[eighty_percent:]
y_test = y[eighty_percent:]

print("\n")
print('X train shape: ', X_train.shape)
print('y train shape: ', y_train.shape)


print("\n")
print('X test shape: ', X_test.shape)
print('y test shape: ', y_test.shape)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.fc1 = nn.Linear(35344, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 28)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 35344)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.dropout1(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
model = NeuralNet()

loss_fn = nn.CrossEntropyLoss()    
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
min_loss = float('inf')

# training
num_epochs = 120

print(len(X_train))

for epoch in range(num_epochs):
    total_correct = 0
    total_loss = 0.0
    
    # Training phase
    model.train()
    for i in range(len(X_train)):
        img = X_train[i].unsqueeze(0)
        label = y_train[i].unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(img)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, prediction = torch.max(outputs.data, 1)
        if prediction.item() == label.item():
            total_correct += 1

    train_loss = total_loss / len(X_train)
    train_accuracy = 100 * (total_correct / len(X_train))
    # Validation phase
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i in range(len(X_test)):
            img = X_test[i].unsqueeze(0)
            label = y_test[i].unsqueeze(0)

            outputs = model(img)
            loss = loss_fn(outputs, label)
            total_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            if prediction.item() == label.item():
                total_correct += 1

    val_loss = total_loss / len(X_test)
    val_accuracy = 100 * (total_correct / len(X_test))
    
    if val_loss < min_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, 'saved_model.pth')
    
    writer.add_scalar("Training Loss/epoch", train_loss, epoch)
    writer.add_scalar("Training Accuracy/epoch", train_accuracy, epoch)
    
    writer.add_scalar("Validation Loss/epoch", val_loss, epoch)
    writer.add_scalar("Validation Accuracy/epoch", val_accuracy, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, \
          Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
writer.flush()
writer.close()
print('Model training complete!')
