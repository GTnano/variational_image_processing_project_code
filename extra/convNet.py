import matplotlib.pyplot as plt
import torch
from torch import nn
from training.data import dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


def resize_image_torch(image, target_shape):
    # Define the transformation
    transform = transforms.Resize(target_shape)
    # Apply the transformation to the image tensor
    resized_image = transform(image)
    return resized_image

def add_white_noise(image, std):
    # Generate white noise with the same shape as the input image tensor
    noise = torch.randn_like(image) * std
    # Add the noise to the image tensor
    noisy_image = image + noise
    return noisy_image


class simple(nn.Module):
    def __init__(self, width, height):
        super(simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        self.avg = nn.AvgPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(56 * 56, 1)
        self.fc2 = nn.Linear(224, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x


model = simple(224, 224)
criterion = nn.MSELoss()

imgs = dataset.H5PY("../training/data/preprocessed/MRI/test.h5")
target_size = (224, 224)
# adding noise
im_sigma = []
y = []
std = [5, 10, 15, 20, 25]
for ix, sigma in enumerate(std):
    tmp = [add_white_noise(img, sigma) for img in imgs]
    # tmp = torch.stack(tmp)
    imgs_reshape = [resize_image_torch(img, target_size) for img in tmp]
    imgs_reshape_fft = [torch.fft.fft2(img) for img in imgs_reshape]
    imgs_tensor = torch.vstack(imgs_reshape_fft)
    im_sigma.append(imgs_tensor)
    y.append(torch.ones(len(im_sigma[ix]))*int(sigma))
x = torch.vstack(im_sigma).view(-1, 1, target_size[0], target_size[1])
y = torch.hstack(y)




dataset = TensorDataset(x, y)

# Define proportions for train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, val_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define batch size and other DataLoader parameters
batch_size = 32
shuffle = True

# Create DataLoader instances for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

train_loss = np.zeros(epochs)
val_loss_arr = np.zeros(epochs)
lr = np.zeros(epochs)
name = 'simple'
model_name = name + '.pth'

mode = 'test'
if mode == 'train':
    model.train()
    print("start training")
    for epoch in range(epochs):
        # Training
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0
        for inputs, labels in val_loader:
            # Forward pass
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        # Update learning rate scheduler
        lr_scheduler.step(val_loss)

        # Print current learning rate
        print("Epoch:", epoch, 'Loss:', loss.detach(), "val loss", val_loss, "Learning Rate:",  optimizer.param_groups[0]['lr'])
        train_loss[epoch] = loss.detach()
        val_loss_arr[epoch] = val_loss
        lr[epoch] = optimizer.param_groups[0]['lr']



    torch.save(model.state_dict(), model_name)

model = simple(224, 224)
model.load_state_dict(torch.load(model_name))
model.eval()
loss = 0
pred = []
y_test = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.float()
        labels = labels.float()
        outputs = model(inputs)
        # pred = torch.argmax(outputs, 1)
        loss += criterion(outputs, labels)
        y_test.append(labels)
        pred.append(outputs)
    print(f"Loss: {loss}")

y_test = torch.concat(y_test)
pred = torch.concat(pred)
sorted_indices = torch.argsort(y_test)
y_test = y_test[sorted_indices]
pred = pred[sorted_indices]
plt.figure()
plt.plot(np.arange(y_test.size(0)), y_test, '*')
plt.plot(np.arange(pred.size(0)), pred, '*')
plt.show()