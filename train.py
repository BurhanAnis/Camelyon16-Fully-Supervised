import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import time
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from Dataset import HistologyTileDataset, create_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=25, validate_every=5):
    since = time.time()
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    print(f"Training on device: {device}")
    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print("-" * 50)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds))
        history['train_loss'].append(epoch_loss)
        history['train_metrics'].append(train_metrics)

        print(f'Train Loss: {epoch_loss:.4f}')
        for k, v in train_metrics.items():
            print(f'  {k}: {v:.4f}')

        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            model.eval()
            running_loss, all_labels, all_preds = 0.0, [], []
            val_pbar = tqdm(val_loader, desc="Validating")

            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            val_loss = running_loss / len(val_loader.dataset)
            val_metrics = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds))
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            print(f'Validation Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  {k}: {v:.4f}')

            save_best_model(val_metrics['f1'], epoch, model, optimizer, val_metrics)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_wts = model.state_dict()

        scheduler.step()

    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'])
    plt.plot(range(0, len(history['val_loss'])*5, 5), history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train', 'Val'])

    metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot([m[metric] for m in history['train_metrics']])
        plt.plot(range(0, len(history['val_metrics'])*5, 5), [m[metric] for m in history['val_metrics']])
        plt.title(metric)
        plt.legend(['Train', 'Val'])

    plt.tight_layout()
    os.makedirs('histology_models', exist_ok=True)
    plt.savefig('histology_models/training_history.png')
    plt.show()

if __name__ == '__main__':
    # Paths
    pos_slide_path = '/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/positive/images'
    neg_slide_path = '/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/negative'
    pos_grid_path = '/Users/burhananis/fully-supervised-camelyon/data/tumour_grid.pkl'
    neg_grid_path = '/Users/burhananis/fully-supervised-camelyon/data/tile_coords_neg.pkl'

    train_loader, val_loader = create_dataloaders(
        pos_slide_path, neg_slide_path,
        pos_grid_path, neg_grid_path,
        batch_size=8, tile_size=256, samples_per_class=2000, num_workers=8
    )

    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    save_best_model = SaveBestModel(metric_name='f1', save_dir='histology_models')

    model, history = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=25, validate_every=5)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, 'histology_models/final_model.pth')

    plot_training_history(history)