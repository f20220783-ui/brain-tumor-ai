import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 1. Read the saved dataset path
with open("dataset_path.txt", "r", encoding="utf-8") as f:
    path = f.read().strip()

# 2. Apply the exact same preprocessing used during training
tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test = datasets.ImageFolder(f"{path}/Testing", tf)
test_dl = torch.utils.data.DataLoader(test, batch_size=32)

# 3. Rebuild ResNet-18 and load your saved weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(test.classes))
model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location=device))
model = model.to(device)

# 4. Evaluate across all 1,600 unseen test scans
correct = total = 0
model.eval()
with torch.no_grad():
    for x, y in test_dl:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)

print(f"accuracy on scans it has never seen: {correct / total:.1%}")