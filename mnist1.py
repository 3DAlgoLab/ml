#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


#%%


class Net(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        dummy = torch.zeros(img_shape).unsqueeze_(0)
        flatten_size = self.cnn_stack(dummy).shape[1]
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def cnn_stack(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.cnn_stack(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: ", device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset_train = datasets.MNIST(
    "./data/mnist", train=True, download=True, transform=transform
)

dataset_test = datasets.MNIST("./data/mnist", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=False)

#%%

print("Train Loader Check:")
for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, y.dtype: {y.dtype}")
    break

print("\nTest Loader Check:")
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, y.dtype: {y.dtype}")
    break

#%%

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.2)
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
loss_fn = nn.CrossEntropyLoss()

#%%


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


#%%
epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
    scheduler.step()

print("Done!")

# %%
