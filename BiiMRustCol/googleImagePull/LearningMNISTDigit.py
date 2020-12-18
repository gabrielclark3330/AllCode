import torch

# part 1 basic inputs with tensors --------------------------------------------------
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
print(x*y)
z = torch.Tensor([3,5,10,15])
print(z)
print(z.view(2,2))
x = torch.zeros([2,5])
print(x)
print(x.shape)

y = torch.rand([2,5])
print(y)

# numpy reshape equivilant = view
y = y.view([1,10])
print(y)

# part 2-----------------------------------------------------------------------------

import torch 
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]))

# A good bach size is between 8 and 64 making it larger helps decrease training time.
# And shuffling is almost allways good for generalization.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

### All over how to view a data set/ calculate percentages/ plot individual matricies

for data in trainset:
    print(data)
    print("only printed the ")
    break

x, y = data[0][0], data[1][0]
print(y)
print(data[0][0].shape)

import matplotlib.pyplot as plt
plt.imshow(data[0][0].view(28,28))
plt.show()

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, Ys = data
    for y in Ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total * 100}")

# part 3 building the neural network architecture----------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim = 1)

net = Net()
print(net)

x = torch.rand((28,28))
x = x.view(-1,28*28)

output = net(x)

print(output)

#part 4 -----------------------------------------------

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        #Data is a bath of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

# check acuracy
correct = 0 
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print(f"acuracy is : %{correct/total * 100}")

import matplotlib.pyplot as plt
plt.imshow(X[0].view(28, 28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))

# Finished This model with decent acuracy!!