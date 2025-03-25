import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from loader import get_dataloaders
from model import get_model

parser = argparse.ArgumentParser(description="Train Model")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--resume-training", action="store_true")
parser.add_argument("--model-name", type=str, default="resnext101")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

train_loader, val_loader, _, train_size, val_size, _ = get_dataloaders(
    "./data", batch_size=args.batch_size
)

model = get_model(num_classes=100, model_name=args.model_name,
                  device=device, dropout=0.3)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3.9e-5, weight_decay=1.5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
scaler = GradScaler()


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc):
    checkpoint_file = os.path.join(save_dir, f"{args.model_name}_checkpoint_e{epoch}.pth")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc
    }, checkpoint_file)

    print(f"Checkpoint saved: {checkpoint_file}")


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=10, resume_training=False):
    start_epoch = 0
    best_acc = 0.0

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"Restarting from {start_epoch} epoch")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        model.train()
        train_loss, correct_top1, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = outputs.topk(1, dim=1)
            correct_top1 += (preds.view(-1) == labels).sum().item()
            total += labels.size(0)
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = correct_top1 / total

        model.eval()
        val_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds_top1 = outputs.topk(1, dim=1)
                _, preds_top5 = outputs.topk(5, dim=1)

                correct_top1 += (preds_top1.view(-1) == labels).sum().item()
                correct_top5 += sum(
                    [labels[i] in preds_top5[i] for i in range(labels.size(0))]
                )
                total += labels.size(0)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc1 = correct_top1 / total
        val_acc5 = correct_top5 / total

        if val_acc1 > best_acc:
            best_acc = val_acc1
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, f"best_{args.model_name}.pth"))
            print(f"Best model saved with Val Acc@1: {val_acc1*100:.2f}%")

        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_acc1*100:.2f}% | Val Acc@5: {val_acc5*100:.2f}%\n")

    print("Done")
    return model


model = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    num_epochs=args.epochs, resume_training=args.resume_training
)