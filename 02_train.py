import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 1. Read the dataset folder path saved in Step 2
with open("dataset_path.txt", "r", encoding="utf-8") as f:
    path = f.read().strip()

# 2. Prepare images so they match what ResNet-18 expects
tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train = datasets.ImageFolder(f"{path}/Training", tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

# 3. Load pretrained ResNet-18 and freeze its existing layers
model = models.resnet18(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad = False  # keep what it already knows

# Replace the final layer to classify our 4 brain MRI categories
model.fc = nn.Linear(model.fc.in_features, len(train.classes))

# 4. Configure device, optimizer, and loss function
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")
model = model.to(device)

opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 5. Train for 5 epochs
for epoch in range(5):
    print(f"Running epoch {epoch + 1}/5...")
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"epoch {epoch + 1}: loss {loss.item():.3f}")

# 6. Save the trained weights to disk for the next steps
torch.save(model.state_dict(), "brain_tumor_resnet18.pth")
print("Training complete! Model saved as brain_tumor_resnet18.pth")