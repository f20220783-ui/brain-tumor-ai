import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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

# 3. Rebuild ResNet-18 and load your saved weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(test.classes))
model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location=device))
model = model.to(device)

# 4. Re-enable gradients so Grad-CAM can trace back to the image
for p in model.parameters():
    p.requires_grad = True
model.eval()

# 5. Pick one scan from Testing (try 0, 500, 900, or 1300 for different classes)
x, y = test[0]

# Check what your model predicts for this scan
with torch.no_grad():
    pred_idx = model(x.unsqueeze(0).to(device)).argmax(1).item()

# 6. Generate the Grad-CAM heatmap on the final convolutional layer
with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
    heat = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]

# 7. Undo normalization so the underlying MRI displays properly
img = x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
overlay = show_cam_on_image(img.clip(0, 1).astype("float32"), heat, use_rgb=True)

plt.imshow(overlay)
plt.title(f"real: {test.classes[y]} | predicted: {test.classes[pred_idx]}")
plt.axis("off")
plt.savefig("gradcam_heatmap.png", bbox_inches="tight")
print("Saved heatmap to gradcam_heatmap.png")
plt.show()