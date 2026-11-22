import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, models, transforms

# 1. Load dataset path and Testing images
with open("dataset_path.txt", "r", encoding="utf-8") as f:
    path = f.read().strip()

tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test = datasets.ImageFolder(f"{path}/Testing", tf)
test_dl = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

# 2. Load your fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(test.classes))
model.load_state_dict(torch.load("brain_tumor_resnet18_finetuned.pth", map_location=device))
model = model.to(device)
model.eval()

# 3. Find all misclassified scans in the Testing set
mistakes = []
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_dl):
        preds = model(x.to(device)).argmax(1).cpu()
        for i in range(len(y)):
            if preds[i] != y[i]:
                global_idx = batch_idx * 32 + i
                mistakes.append((global_idx, y[i].item(), preds[i].item()))

print(f"Found {len(mistakes)} total mistakes out of {len(test)} scans.")

# 4. Enable gradients and plot Grad-CAM heatmaps for the first 4 mistakes
for p in model.parameters():
    p.requires_grad = True

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
    for ax, (idx, true_lbl, pred_lbl) in zip(axes.flat, mistakes[:4]):
        x, _ = test[idx]
        heat = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]

        img = x.permute(1, 2, 0).numpy() * std + mean
        overlay = show_cam_on_image(img.clip(0, 1).astype("float32"), heat, use_rgb=True)

        ax.imshow(overlay)
        ax.set_title(f"real: {test.classes[true_lbl]} | pred: {test.classes[pred_lbl]}")
        ax.axis("off")

plt.tight_layout()
plt.savefig("error_heatmaps.png", bbox_inches="tight")
print("Saved 4 mistake heatmaps to error_heatmaps.png")
plt.show()