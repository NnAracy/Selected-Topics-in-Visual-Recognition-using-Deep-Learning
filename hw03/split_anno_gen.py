import os
import json
import shutil
import numpy as np
from PIL import Image
from skimage import measure
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio
import pycocotools.mask as mask_utils


def generate_coco_annotations(train_dir, output_json):
    """Generate COCO format annotations for the given directory."""
    categories = [
        {"id": 1, "name": "class1"},
        {"id": 2, "name": "class2"},
        {"id": 3, "name": "class3"},
        {"id": 4, "name": "class4"}
    ]

    images = []
    annotations = []
    image_id = 1
    ann_id = 1
    error_files = []

    for folder in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        img_path = os.path.join(folder_path, "image.tif")
        if not os.path.exists(img_path):
            error_files.append(img_path)
            continue

        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            error_files.append(img_path)
            print(f"Error reading {img_path}: {e}")
            continue

        images.append({
            "id": image_id,
            "file_name": os.path.join(folder, "image.tif"),
            "width": width,
            "height": height
        })

        for class_id, class_name in enumerate(
            ["class1", "class2", "class3", "class4"], 1
        ):
            mask_path = os.path.join(folder_path, f"{class_name}.tif")
            if not os.path.exists(mask_path):
                continue

            try:
                mask = imageio.imread(mask_path)
                mask = (mask > 0).astype(np.uint8)
            except Exception as e:
                error_files.append(mask_path)
                print(f"Error reading {mask_path}: {e}")
                continue

            labeled_mask = measure.label(mask)
            for region_id in np.unique(labeled_mask)[1:]:
                binary_mask = (labeled_mask == region_id).astype(np.uint8)

                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                area = int(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id += 1

        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    if error_files:
        print("Errors encountered with the following files:")
        for f in error_files:
            print(f)
    else:
        print(f"Successfully generated {output_json}")


def split_and_copy_data(data_root, train_ratio=0.8):
    """Split dataset into train and val by copying directories and generating annotations."""
    train_origin_dir = os.path.join(data_root, "train_origin")
    original_train_dir = os.path.join(data_root, "train")
    new_train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Check if train_origin exists; if not, rename train to train_origin
    if not os.path.exists(train_origin_dir):
        if not os.path.exists(original_train_dir):
            raise FileNotFoundError(
                f"Neither {train_origin_dir} nor {original_train_dir} found"
            )
        print(f"Renaming {original_train_dir} to {train_origin_dir}")
        os.rename(original_train_dir, train_origin_dir)
    else:
        print(f"Using existing {train_origin_dir}")

    # Get all subdirectories from train_origin
    folders = [
        f for f in os.listdir(train_origin_dir)
        if os.path.isdir(os.path.join(train_origin_dir, f))
    ]
    if not folders:
        raise ValueError(f"No subdirectories found in {train_origin_dir}")

    print(f"Found {len(folders)} subdirectories in {train_origin_dir}")

    # Split folders
    train_folders, val_folders = train_test_split(
        folders, train_size=train_ratio, random_state=42
    )
    print(f"Train folders: {len(train_folders)}, Val folders: {len(val_folders)}")

    # Remove existing train and val directories
    if os.path.exists(new_train_dir):
        print(f"Removing existing {new_train_dir}")
        shutil.rmtree(new_train_dir)

    if os.path.exists(val_dir):
        print(f"Removing existing {val_dir}")
        shutil.rmtree(val_dir)

    # Create new directories
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy train folders
    for folder in train_folders:
        src_folder = os.path.join(train_origin_dir, folder)
        dst_folder = os.path.join(new_train_dir, folder)
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        print(f"Copied {src_folder} to {dst_folder}")

    # Copy val folders
    for folder in val_folders:
        src_folder = os.path.join(train_origin_dir, folder)
        dst_folder = os.path.join(val_dir, folder)
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        print(f"Copied {src_folder} to {dst_folder}")

    # Generate annotations
    generate_coco_annotations(new_train_dir, os.path.join(data_root, "train.json"))
    generate_coco_annotations(val_dir, os.path.join(data_root, "val.json"))


def main():
    data_root = "./hw3-data-release"
    split_and_copy_data(data_root)


if __name__ == "__main__":
    main()