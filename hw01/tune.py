import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import argparse
from tqdm import tqdm
from loader import get_dataloaders
from model import get_model

parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--trials", type=int, default=20)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1
print(f"Using {torch.cuda.device_count()} GPUs")


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 5e-4)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64])

    train_loader, val_loader, _, train_size, val_size, _ = get_dataloaders(
        "./data", batch_size=batch_size
    )

    model = get_model(num_classes=100, model_name="resnet101",
                      device=device, dropout=dropout)

    if multi_gpu:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
)
study.optimize(objective, n_trials=args.trials)

best_params = study.best_params
print(f"Optuna best hyperparameters: {best_params}")
print(f"Best validation accuracy: {study.best_value:.2f}%")

with open("./best_hyperparams.txt", "w") as f:
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Best Validation Accuracy: {study.best_value:.2f}%\n")