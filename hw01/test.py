import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import glob
from tqdm import tqdm

# 自定義測試數據集
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))  # 讀取所有 .jpg 文件
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path

# 設定參數
train_dir = "./data/train"  # 替換為你的訓練資料夾路徑
test_dir = "./data/test"    # 替換為你的測試資料夾路徑
save_dir = "."
save_path = os.path.join(save_dir, "prediction.csv")
model_path = "./best/best_resnext50.pth"
batch_size = 64
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. 獲取訓練時的類別映射並修正為數值順序
train_dataset = datasets.ImageFolder(train_dir)
default_class_to_idx = train_dataset.class_to_idx  # 字母順序的映射
numeric_class_to_idx = {str(i): i for i in range(100)}  # 數值順序映射: {'0': 0, '1': 1, ..., '99': 99}
idx_remap = {default_class_to_idx[k]: numeric_class_to_idx[k] for k in default_class_to_idx.keys()}  # 字母索引 -> 數值索引

# 2. 加載測試數據
test_dataset = TestDataset(test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 3. 加載 ResNet101 模型
model = models.resnext50_32x4d(pretrained=False)  # 注意：這裡假設模型結構與訓練時一致
num_classes = 100  # 假設有 100 個類別
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 替換最後一層
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# 4. 進行測試並收集結果
results = []
with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.topk(1, dim=1)  # 取 Top-1 預測結果

        for path, pred in zip(paths, preds.view(-1).cpu().numpy()):
            image_name = os.path.basename(path)  # 獲取文件名（去除路徑）
            image_name = os.path.splitext(image_name)[0]  # 去除 `.jpg` 後綴
            remapped_pred = idx_remap[pred]  # 修正預測索引
            results.append((image_name, remapped_pred))

# 5. 儲存為 CSV
df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv(save_path, index=False)

print(f"測試結果已保存至: {save_path}！")
