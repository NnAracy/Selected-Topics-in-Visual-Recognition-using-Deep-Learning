import os
import glob
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Image Classification Prediction")
parser.add_argument("--model-name", type=str, default="resnext50")
args = parser.parse_args()


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
model_path = f"./best/best_{args.model_name}.pth"
batch_size = 64
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir)
default_class_to_idx = train_dataset.class_to_idx
numeric_class_to_idx = {str(i): i for i in range(100)}
idx_remap = {
    default_class_to_idx[k]: numeric_class_to_idx[k]
    for k in default_class_to_idx.keys()
}

test_dataset = TestDataset(test_dir, transform=val_transforms)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    shuffle=False, num_workers=num_workers
)

model_dict = {
    "resnext50": models.resnext50_32x4d,
    "resnet101": models.resnet101,
    "resnext101": models.resnext101_32x8d
}

if args.model_name not in model_dict:
    raise ValueError(f"Unsupported model name: {args.model_name}")

model = model_dict[args.model_name](pretrained=False)
num_classes = 100
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

results = []
with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.topk(1, dim=1)

        for path, pred in zip(paths, preds.view(-1).cpu().numpy()):
            image_name = os.path.basename(path)
            image_name = os.path.splitext(image_name)[0]
            remapped_pred = idx_remap[pred]
            results.append((image_name, remapped_pred))

df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv(save_path, index=False)

print(f"Prediction saved at: {save_path}！")