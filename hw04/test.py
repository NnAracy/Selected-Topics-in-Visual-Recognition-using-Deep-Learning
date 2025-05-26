import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from collections import OrderedDict

from dataset import PromptIRDataset
from model import PromptIR


def main(args):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    # --- Dataset and DataLoader ---
    test_dataset = PromptIRDataset(
        root_dir=args.data_root,
        mode='test',
        patch_size=None
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # --- Model ---
    model = PromptIR(dim=args.model_dim, num_blocks=args.num_blocks, heads=args.heads, decoder=True).to(device)
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    print(f"=> Loading checkpoint for testing from '{args.model_path}'")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model_weights_to_load = checkpoint['model_state_dict']
        print("Loaded 'model_state_dict' from comprehensive checkpoint.")
    else:
        model_weights_to_load = checkpoint
        print("Warning: Loaded checkpoint does not seem to be a comprehensive training checkpoint. "
              "Assuming it's a model state_dict directly.")
    cleaned_state_dict = OrderedDict()
    has_module_prefix = False
    for k, v in model_weights_to_load.items():
        if k.startswith('module.'):
            has_module_prefix = True
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    if has_module_prefix:
        print("Removed 'module.' prefix from model state_dict keys.")
        model.load_state_dict(cleaned_state_dict)
    else:
        model.load_state_dict(model_weights_to_load)
    model.eval()
    print(f"Loaded model from {args.model_path}")
    # --- Inference Loop ---
    with torch.no_grad():
        for i, (degraded_batch, filenames) in enumerate(test_loader):
            if degraded_batch is None:
                print(f"Skipping due to loading error for an image associated with {filenames}")
                continue
            degraded_batch = degraded_batch.to(device)
            filename = filenames[0]
            restored_batch = model(degraded_batch)
            output_path = os.path.join(args.output_dir, filename)
            restored_image_tensor = torch.clamp(restored_batch, 0, 1)
            save_image(restored_image_tensor, output_path)
            print(f"Processed and saved: {output_path}")
    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptIR Inference')
    parser.add_argument('--data_root', type=str, default='./hw4_realse_dataset',
                        help='Path to dataset root')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./results_promptir',
                        help='Directory to save restored images')
    parser.add_argument('--device', type=str, default='cuda', choices=['gpu', 'cpu'], help='Device')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size images were trained on / target inference size. '
                             'Use None if processing full images of variable size '
                             '(model must support this).')
    parser.add_argument('--model_dim', type=int, default=48, help='Model dimension (PromptIR dim)')
    parser.add_argument('--num_blocks', type=list, default=[4, 6, 6, 8],
                        help='Number of blocks per encoder level (PromptIR)')
    parser.add_argument('--heads', type=list, default=[1, 2, 4, 8],
                        help='Number of attention heads per level (PromptIR)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    if isinstance(args.patch_size, str) and args.patch_size.lower() == 'none':
        args.patch_size = None
    main(args)