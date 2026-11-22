import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 1. Read the saved dataset path
with open("dataset_path.txt", "r", encoding="utf-8") as f:
    path = f.read().strip()

# 2. Apply the same preprocessing pipeline
tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test = datasets.ImageFolder(f"{path}/Testing", tf)
test_dl = torch.utils.data.DataLoader(test, batch_size=32)

# 3. Rebuild ResNet-18 and load your NEW fine-tuned weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(test.classes))
model.load_state_dict(torch.load("brain_tumor_resnet18_finetuned.pth", map_location=device))
model = model.to(device)

# 4. Score overall and per-class accuracy on unseen test scans
correct = total = 0
class_correct = [0] * len(test.classes)
class_total = [0] * len(test.classes)

model.eval()
with torch.no_grad():
    for x, y in test_dl:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)

        for label, prediction in zip(y, pred):
            class_total[label] += 1
            if label == prediction:
                class_correct[label] += 1

print(f"New accuracy on unseen scans: {correct / total:.1%} (baseline was 84.0%)\n")
for idx, class_name in enumerate(test.classes):
    acc = class_correct[idx] / class_total[idx]
    print(f"  {class_name:12s}: {class_correct[idx]}/{class_total[idx]} ({acc:.1%})")