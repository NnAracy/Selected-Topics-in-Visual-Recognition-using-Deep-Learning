import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from schedulers import LinearWarmupCosineAnnealingLR
from ssim_loss import SSIMLoss
import torch.distributed as dist
from collections import OrderedDict

from dataset import PromptIRDataset
from model import PromptIR


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=args.world_size, rank=args.rank)
        setup_for_distributed(args.rank == 0)
        print(f"Distributed training initialized. World size: {args.world_size}, "
              f"Rank: {args.rank}, Local GPU: {args.gpu}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        if torch.cuda.is_available():
            args.gpu = 0
            torch.cuda.set_device(args.gpu)
        else:
            args.gpu = None
        return


def main(args):
    init_distributed_mode(args)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None and
                          torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    scanner_dset = PromptIRDataset(root_dir=args.data_root, mode='train',
                                   patch_size=args.patch_size)
    all_paired_files = scanner_dset.image_files
    if not all_paired_files:
        print(f"Error: No training files found in {os.path.join(args.data_root, 'train')}. "
              f"Exiting.")
        return
    train_files, val_files = train_test_split(all_paired_files, test_size=args.val_split,
                                              random_state=args.seed)
    train_dataset = PromptIRDataset(
        root_dir=args.data_root,
        mode='train',
        file_list=train_files,
        patch_size=args.patch_size,
    )
    val_dataset = PromptIRDataset(
        root_dir=args.data_root,
        mode='val',
        file_list=val_files,
        patch_size=args.patch_size
    )
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size,
                                          rank=args.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size,
                                        rank=args.rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler
    )
    # --- Model ---
    model = PromptIR(dim=args.model_dim, num_blocks=args.num_blocks, heads=args.heads,
                     decoder=True).to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                   find_unused_parameters=True)
    elif torch.cuda.device_count() > 1 and not args.distributed:
        print(f"Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    # --- Loss, Optimizer, Scheduler ---
    criterion = SSIMLoss(data_range=1.0, channel=3, size_average=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15,
                                             max_epochs=150)
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print(f"=> Loading checkpoint '{args.resume_checkpoint}'")
            if device.type == 'cuda':
                checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            else:
                checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            model_to_load = model.module if args.distributed or isinstance(model, nn.DataParallel) else model
            saved_state_dict = checkpoint['model_state_dict']
            current_model_is_wrapped_for_loading = args.distributed or isinstance(model, nn.DataParallel)
            saved_keys_have_module_prefix = any(key.startswith('module.') for key in
                                                saved_state_dict.keys())
            if saved_keys_have_module_prefix and not current_model_is_wrapped_for_loading:
                new_state_dict_for_load = OrderedDict()
                for k, v in saved_state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict_for_load[k[7:]] = v
                    else:
                        new_state_dict_for_load[k] = v
                model_to_load.load_state_dict(new_state_dict_for_load)
            elif not saved_keys_have_module_prefix and current_model_is_wrapped_for_loading:
                model_to_load.load_state_dict(saved_state_dict)
            else:
                model_to_load.load_state_dict(saved_state_dict)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            except KeyError as e:
                print(f"Warning: Checkpoint missing key {e}. Skipping restore for that component.")
            print(f"=> Loaded checkpoint '{args.resume_checkpoint}' "
                  f"(trained for {checkpoint['epoch']+1} epochs)")
            print(f"=> Resuming training from epoch {start_epoch}")
            print(f"=> Best validation loss from checkpoint: {best_val_loss:.4f}")
            seed = args.seed + args.rank + start_epoch
        else:
            print(f"=> No checkpoint found at '{args.resume_checkpoint}'. Starting from scratch.")
            seed = args.seed + args.rank
    else:
        seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # --- Training Loop ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_train_loss = 0.0
        for i, (degraded_batch, clean_batch) in enumerate(train_loader):
            degraded_batch, clean_batch = degraded_batch.to(device), clean_batch.to(device)
            optimizer.zero_grad()
            restored_batch = model(degraded_batch)
            loss = criterion(restored_batch, clean_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if args.rank == 0 and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Train Loss: {loss.item():.4f}")
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        if args.rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Avg Train Loss: {avg_epoch_train_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        # --- Validation Loop ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for degraded_batch, clean_batch in val_loader:
                degraded_batch, clean_batch = degraded_batch.to(device), clean_batch.to(device)
                restored_batch = model(degraded_batch)
                loss = criterion(restored_batch, clean_batch)
                epoch_val_loss += loss.item()
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        if args.distributed:
            val_loss_tensor = torch.tensor(avg_epoch_val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_val_loss = val_loss_tensor.item() / args.world_size
        if args.rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Avg Validation Loss: {avg_epoch_val_loss:.4f}")
            model_state_to_save = model.module.state_dict() if args.distributed or \
                isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args
            }
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                checkpoint_data['best_val_loss'] = best_val_loss
                save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save(checkpoint_data, save_path)
                print(f"Best model checkpoint saved to {save_path} "
                      f"(Epoch: {epoch+1}, Val Loss: {best_val_loss:.4f})")
            if (epoch + 1) % 10 == 0:
                epoch_specific_save_path = os.path.join(args.checkpoint_dir,
                                                       f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint_data, epoch_specific_save_path)
                print(f"Saved checkpoint for epoch {epoch+1} to {epoch_specific_save_path}")
            fixed_latest_save_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
            torch.save(checkpoint_data, fixed_latest_save_path)
            print(f"Latest checkpoint marker updated at {fixed_latest_save_path} "
                  f"(Epoch: {epoch+1})")
        scheduler.step()
    if args.rank == 0:
        print("Training complete.")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptIR Training')
    parser.add_argument('--data_root', type=str, default='./hw4_realse_dataset',
                        help='Path to dataset')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for training')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_dim', type=int, default=48,
                        help='Model dimension (PromptIR dim)')
    parser.add_argument('--num_blocks', type=list, default=[4, 6, 6, 8],
                        help='Number of blocks per encoder level (PromptIR)')
    parser.add_argument('--heads', type=list, default=[1, 2, 4, 8],
                        help='Number of attention heads per level (PromptIR)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size (per GPU)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_promptir',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Interval for logging training batch loss')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from '
                             '(e.g., ./checkpoints_promptir/latest_checkpoint.pth)')
    # DDP arguments
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training (set by launcher)')
    args = parser.parse_args()
    main(args)