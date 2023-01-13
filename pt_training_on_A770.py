import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time

# Set device & Hyperparameters
device = torch_directml.device()
num_classes = 196   # The Cars dataset contains 16,185 images of 196 classes of cars
learning_rate = 1e-3
batch_size = 32

# Step1: Load Flower102 dataset
# https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html
data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.StanfordCars(root="dataset/", split="train", transform=data_transforms["train"], download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.StanfordCars(root="dataset/", split='test', transform=data_transforms["test"], download=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step2: Load Pretrained ResNet50 and add FC layer
model = models.resnet50(weights='DEFAULT').to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 256),
               nn.ReLU(inplace=True),
               nn.Linear(256, num_classes)).to(device)
model.train()
# Step4: define Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step5: Train Network
for epoch in range(3):
    losses=[]
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        start_time = time.time()
        data = data.to(device)
        targets = targets.to(device)

        # forward
        preds = model(data)
        loss = loss_fn(preds, targets)
        losses.append(loss)
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # GSD
        optimizer.step()
        time_elapsed = time.time() - start_time
        print(f"Step:{batch_idx}, elapsed time: {time_elapsed*1000:0.2f}ms; loss is {sum(losses)/len(losses)}.")

# Step6: Chekc accuracy on test dataset
num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
    for data, targets in test_dataloader:
        data = data.to(device)
        targets = targets.to(device)
        # data = data.reshape(data.shape[0], -1)

        preds = model(data)
        _, results = preds.max(1)
        # print(preds.shape, results.shape, targets.shape)
        num_correct += (results == targets).sum()
        num_samples += results.size(0)
    print(f"The accuracy on test dataset is : {float(num_correct)/float(num_samples)*100:.2f}%")