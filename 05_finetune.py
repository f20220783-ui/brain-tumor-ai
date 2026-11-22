import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 1. Read the saved dataset path
with open("dataset_path.txt", "r", encoding="utf-8") as f:
    path = f.read().strip()

# 2. Same preprocessing pipeline as before
tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train = datasets.ImageFolder(f"{path}/Training", tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

# 3. Rebuild ResNet-18 and load your 84% baseline weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(train.classes))
model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location=device))

# 4. Unfreeze ALL layers so the entire network can adapt to MRI scans
for p in model.parameters():
    p.requires_grad = True

model = model.to(device)
print(f"Fine-tuning entire model on: {device}")

# 5. Update ALL parameters using a 10x smaller learning rate (1e-4)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# 6. Train for 4 fine-tuning epochs
for epoch in range(4):
    print(f"\nRunning fine-tune epoch {epoch + 1}/4...")
    model.train()
    for batch_idx, (x, y) in enumerate(train_dl, start=1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(train_dl)} - loss: {loss.item():.3f}")

    print(f"epoch {epoch + 1} finished: loss {loss.item():.3f}")

# 7. Save the fine-tuned model separately so your baseline stays safe
torch.save(model.state_dict(), "brain_tumor_resnet18_finetuned.pth")
print("\nFine-tuning complete! Saved as brain_tumor_resnet18_finetuned.pth")