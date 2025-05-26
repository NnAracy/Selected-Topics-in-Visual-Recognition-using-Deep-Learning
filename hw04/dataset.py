import os
import glob
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def data_augmentation(image_np, mode):
    if mode == 0:
        # original
        out = image_np
    elif mode == 1:
        # flip up and down
        out = np.flipud(image_np)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image_np)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image_np)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image_np, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image_np, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image_np, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image_np, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.ascontiguousarray(out)


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data_np in args:
        out.append(data_augmentation(data_np, flag_aug).copy())
    return out


def crop_img_to_base(img_np, base=16):
    h, w, _ = img_np.shape
    new_h = (h // base) * base
    new_w = (w // base) * base
    if new_h == h and new_w == w:
        return img_np
    return img_np[:new_h, :new_w, :]


def random_crop_paired_numpy(img1_np, img2_np, patch_size):
    h, w, _ = img1_np.shape
    if h < patch_size or w < patch_size:
        ph, pw = min(h, patch_size), min(w, patch_size)
        return img1_np[:ph, :pw, :], img2_np[:ph, :pw, :]
    rand_h = random.randint(0, h - patch_size)
    rand_w = random.randint(0, w - patch_size)
    patch1 = img1_np[rand_h:rand_h + patch_size, rand_w:rand_w + patch_size, :]
    patch2 = img2_np[rand_h:rand_h + patch_size, rand_w:rand_w + patch_size, :]
    return patch1, patch2


class PromptIRDataset(Dataset):
    def __init__(self, root_dir, mode='train', file_list=None, patch_size=256,
                 test_degraded_subdir='degraded'):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.patch_size = patch_size
        self.image_files = []
        if file_list:
            self.image_files = file_list
        else:
            if self.mode in ['train', 'val']:
                self._scan_paired_files()
            elif self.mode == 'test':
                self._scan_test_files(test_degraded_subdir)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'val', 'test'.")
        self.to_tensor = transforms.ToTensor()  # Scales images to [0, 1]

    def _scan_paired_files(self):
        degraded_base = os.path.join(self.root_dir, 'train', 'degraded')
        clean_base = os.path.join(self.root_dir, 'train', 'clean')
        for type_prefix in ["rain", "snow"]:
            degraded_paths = glob.glob(os.path.join(degraded_base, f"{type_prefix}-*.png"))
            for deg_path in degraded_paths:
                filename = os.path.basename(deg_path)
                try:
                    img_id_str = filename.split('-')[1].split('.')[0]
                    cln_path = os.path.join(clean_base, f"{type_prefix}_clean-{img_id_str}.png")
                    if os.path.exists(cln_path):
                        self.image_files.append({'degraded': deg_path, 'clean': cln_path})
                except IndexError:
                    pass

    def _scan_test_files(self, test_degraded_subdir):
        test_degraded_path = os.path.join(self.root_dir, 'test', test_degraded_subdir)
        if not os.path.isdir(test_degraded_path):
            print(f"Warning: Test directory not found: {test_degraded_path}")
            return
        img_names = os.listdir(test_degraded_path)
        png_files = [f for f in img_names if f.lower().endswith('.png')]
        sorted_png_files = sorted(png_files, key=lambda x: int(os.path.splitext(x)[0]))
        for img_name in sorted_png_files:
            self.image_files.append(os.path.join(test_degraded_path, img_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            paths = self.image_files[idx]
            try:
                degraded_pil = Image.open(paths['degraded']).convert('RGB')
                clean_pil = Image.open(paths['clean']).convert('RGB')
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found. Degraded: {paths['degraded']} or Clean: {paths['clean']}")
            degraded_np = np.array(degraded_pil)
            clean_np = np.array(clean_pil)
            degraded_np = crop_img_to_base(degraded_np, base=16)
            clean_np = crop_img_to_base(clean_np, base=16)
            if degraded_np.shape[0] < self.patch_size or degraded_np.shape[1] < self.patch_size:
                h, w, _ = degraded_np.shape
                scale_factor = 1.0
                if h < self.patch_size:
                    scale_factor = max(scale_factor, self.patch_size * 1.1 / h)
                if w < self.patch_size:
                    scale_factor = max(scale_factor, self.patch_size * 1.1 / w)
                if scale_factor > 1.0:
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    degraded_pil_resized = degraded_pil.resize((new_w, new_h), Image.BICUBIC)
                    clean_pil_resized = clean_pil.resize((new_w, new_h), Image.BICUBIC)
                    degraded_np = np.array(degraded_pil_resized)
                    clean_np = np.array(clean_pil_resized)
                    degraded_np = crop_img_to_base(degraded_np, base=16)
                    clean_np = crop_img_to_base(clean_np, base=16)
            degraded_patch_np, clean_patch_np = random_crop_paired_numpy(
                degraded_np, clean_np, self.patch_size
            )
            if self.mode == 'train':
                augmented_patches = random_augmentation(degraded_patch_np, clean_patch_np)
                degraded_patch_np = augmented_patches[0]
                clean_patch_np = augmented_patches[1]
            degraded_tensor = self.to_tensor(degraded_patch_np)
            clean_tensor = self.to_tensor(clean_patch_np)
            return degraded_tensor, clean_tensor
        elif self.mode == 'test':
            degraded_path = self.image_files[idx]
            degraded_pil = Image.open(degraded_path).convert('RGB')
            img_filename = os.path.basename(degraded_path)
            degraded_np = np.array(degraded_pil)
            degraded_np = crop_img_to_base(degraded_np, base=16)
            if self.patch_size is not None:
                pil_img_for_resize = Image.fromarray(degraded_np)
                resized_pil = pil_img_for_resize.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                degraded_tensor_input = self.to_tensor(np.array(resized_pil))
            else:
                degraded_tensor_input = self.to_tensor(degraded_np)
            return degraded_tensor_input, img_filename
        raise RuntimeError(f"getitem called with invalid mode: {self.mode} or index: {idx}")