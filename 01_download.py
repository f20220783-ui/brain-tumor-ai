import os
import kagglehub

# Download the dataset locally
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Dataset downloaded to:", path)

# Save the folder path so our next scripts can find the images automatically
with open("dataset_path.txt", "w", encoding="utf-8") as f:
    f.write(path)

# Verify the subfolders inside Training and Testing
print("Training folders:", sorted(os.listdir(f"{path}/Training")))
print("Testing folders:", sorted(os.listdir(f"{path}/Testing")))