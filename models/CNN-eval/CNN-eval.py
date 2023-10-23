import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import psutil
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_name')

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # return in MB

# Before processing
memory_before = get_memory_usage()
# Start the clock
start_time = time.time()

class SingerDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        label = torch.tensor(label_list, dtype=torch.long)
        self.label_list = label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        return self.data_list[idx], self.label_list[idx]


class SingerClassifier(nn.Module):
    def __init__(self, num_singers):
        super(SingerClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Adjust the FC layer input dimensions based on the output of your conv/pool layers.
        # For this example, I've left a placeholder size.
        self.fc_input_size =16416 #32 * 514 * 108  # Adjust this value based on the convolution and pooling operations

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Final softmax layer
        self.fc3 = nn.Linear(64, num_singers)

    def forward(self, x):
        # Add a channel dimension (B x C x H x W)
        x = x.unsqueeze(1)

        # Convolutional layers with batch normalization and ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling
        x = self.maxpool(x)

        # Average pooling over time
        x = F.avg_pool2d(x, (1, x.size(3)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Softmax layer for classification
        x = F.log_softmax(self.fc3(x), dim=1)
        torch.cuda.empty_cache()
        return x


def evaluate(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss


# Hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 8  # Depending on your data loader and dataset size
num_singer = 2

# Assuming your data is loaded into DataLoader objects
# train_loader, val_loader = ...

# Initialize the model, loss, and optimizer
model = SingerClassifier(num_singers=num_singer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

data_run = 'run8'
X = load('./../data-preprocess/'+data_run+'/data.joblib')
Y = load('./../data-preprocess/'+data_run+'/labels.joblib')


# Find indices where Y is 0 or 1
filtered_indices = np.where((Y == 0) | (Y == 1))[0]

# Use these indices to filter X and Y
X = X[filtered_indices]
Y = Y[filtered_indices]



X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=1/9, random_state=42)

train_dataset = SingerDataset(X_train, Y_train)
val_dataset = SingerDataset(X_val, Y_val)
test_dataset = SingerDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load
loading = False
if loading :
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    prev_loss = checkpoint['loss']

model.train()  # Set the model back to training mode

# Training loop
loss_values = []
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        """
        print("Min label:", labels.min().item())
        print("Max label:", labels.max().item())
        """

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    writer.add_scalar('Loss/train',average_loss, epoch)
    loss_values.append(average_loss)


    # Optionally: Validation and model saving based on performance
    # ...

print('Finished Training')

# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss_values': loss_values  # save the list of loss values here
}, 'checkpoint.pth')
"""
checkpoint = torch.load('checkpoint')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss_values = checkpoint['loss_values']  # load the list of loss values here
"""

criterion = torch.nn.CrossEntropyLoss()  # or whichever loss function you're using

val_loss = evaluate(model, val_loader, criterion, device)
print(f'Validation Loss: {val_loss:.4f}')

test_loss = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}')


memory_after = get_memory_usage()

print(f"Memory used: {memory_after - memory_before} MB")
# End the clock
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")
