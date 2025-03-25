import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path


train_dir = "./data/train"
test_dir = "./data/test"
save_dir = "."
save_path = os.path.join(save_dir, "prediction.csv")
checkpoints = {
    "resnext50": "./best/best_resnext50.pth",
    "resnet101": "./best/best_resnet101.pth",
    "resnext101": "./best/best_resnext101.pth"
}
batch_size = 64
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 100

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir)
default_class_to_idx = train_dataset.class_to_idx
numeric_class_to_idx = {str(i): i for i in range(num_classes)}
idx_remap = {
    default_class_to_idx[k]: numeric_class_to_idx[k]
    for k in default_class_to_idx.keys()
}

test_dataset = TestDataset(test_dir, transform=val_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

models_dict = {
    "resnext50": models.resnext50_32x4d(pretrained=False),
    "resnet101": models.resnet101(pretrained=False)
}

for name, model in models_dict.items():
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoints[name]))
    model = model.to(device)
    model.eval()
    print(f"Loaded {name} from {checkpoints[name]}")

results = []

with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="Predicting with Bagging"):
        images = images.to(device)
        all_probs = torch.zeros(
            len(images), num_classes, len(models_dict)
        ).to(device)

        for i, (name, model) in enumerate(models_dict.items()):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs[:, :, i] = probs

        avg_probs = all_probs.mean(dim=2)
        _, preds = avg_probs.topk(1, dim=1)

        for path, pred in zip(paths, preds.view(-1).cpu().numpy()):
            image_name = os.path.basename(path)
            image_name = os.path.splitext(image_name)[0]
            remapped_pred = idx_remap[pred]
            results.append((image_name, remapped_pred))

df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv(save_path, index=False)

print(f"Prediction saved at: {save_path}！")