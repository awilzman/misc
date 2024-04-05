import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch, network, optimizer, train_losses):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(network, test_losses, test_accuracies):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to GPU
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

# Set seed for reproducibility
seed = 12
torch.manual_seed(seed)

# Define hyperparameters
n_epochs = 10
batch_size_train = 256
batch_size_test = 1000
learning_rate = 1e-3
momentum = 0.5
log_interval = 200

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transform),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True, transform=transform),
    batch_size=batch_size_test, shuffle=True)

# Initialize network and move it to GPU
network = Net().to(device)

# Initialize optimizers with different settings
optimizer_1 = optim.SGD(network.parameters(), lr=learning_rate)
optimizer_2 = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer_3 = optim.Adam(network.parameters(), lr=learning_rate/10)

# Initialize lists to store test metrics
results = []

# Train and test networks
for optimizer, label in zip([optimizer_1, optimizer_2, optimizer_3], ['SGD', 'SGD with Momentum', 'Adam']):
    print(f'\nTraining with {label} optimizer:')
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, optimizer, train_losses)
        test(network, test_losses, test_accuracies)
    # Append test metrics to results list
    results.append({
        'Optimizer': label,
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Test Accuracy (%)': test_accuracies
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('test_metrics.csv', index=False)

#%%
for result in results:
    optimizer = result['Optimizer']
    test_losses = result['Train Loss']
    plt.plot(test_losses, marker='o', label=f'{optimizer} Test Loss')

plt.xlabel('Batch')
plt.ylabel('Train Loss')
plt.title('Train Losses of Different Optimizers')
plt.legend()
plt.show()

for result in results:
    optimizer = result['Optimizer']
    test_losses = result['Test Loss']
    plt.plot(test_losses, marker='o', label=f'{optimizer} Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Losses of Different Optimizers')
plt.legend()
plt.show()
