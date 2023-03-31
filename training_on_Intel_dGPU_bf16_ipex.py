import torch
import torchvision
import intel_extension_for_pytorch as ipex
from tqdm import tqdm

LR = 0.001
DOWNLOAD = True
DATA = 'datasets/food101/'
epochs = 30
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' 

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.Food101(
        root=DATA,
        split='train',
        transform=transform,
        download=DOWNLOAD,
)

val_dataset = torchvision.datasets.Food101(
        root=DATA,
        split='test',
        transform=transform,
        download=DOWNLOAD,
)

train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128
)

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=32
)

model = torchvision.models.resnet50(weights='IMAGENET1K_V2',num_classes=101)
model = model.to('xpu')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)


for epoch in range(epochs):
	tloss,vloss = 0.0, 0.0
	top1,top5 = 0.0, 0.0
	pbar = tqdm(enumerate(train_loader),total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
	for i, (data, target) in pbar:
		model.train()
		data = data.to('xpu')
		target = target.to('xpu')
		with torch.xpu.amp.autocast():
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
                optimizer.step()
		optimizer.zero_grad()
		tloss = (tloss*i + loss.item()) / (i+1)
		pbar.desc = f"{f'epoch:{epoch + 1}/{epochs}'}   train_loss:{tloss:>.3g}  val_loss:{vloss:>.3g}  top1_acc:{top1:>.3g}  top5_acc:{top5:>.3g}" 
		if i == len(pbar) - 1:
			pred,targets,vloss = [], [], 0
			n = len(val_loader)
			model.eval()
			with torch.xpu.amp.autocast():
				for d, (images, labels) in enumerate(val_loader):
					images = images.to('xpu') 
					labels = labels.to('xpu')
					y = model(images)
					pred.append(y.argsort(1, descending=True)[:, :5])
					targets.append(labels) 
					vloss += criterion(y, labels).item()
			vloss /= n
			pred, targets = torch.cat(pred), torch.cat(targets)
			correct = (targets[:, None] == pred).float()
			acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
			top1, top5 = acc.mean(0).tolist()
			pbar.desc = f"{f'epoch:{epoch + 1}/{epochs}'}   train_loss:{tloss:>.3g}  val_loss:{vloss:>.3g}  top1_acc:{top1:>.3g}  top5_acc:{top5:>.3g}" 

	torch.save({
	     'model_state_dict': model.state_dict(),
	     'optimizer_state_dict': optimizer.state_dict(),
	     }, f'{epoch}_checkpoint.pth')
