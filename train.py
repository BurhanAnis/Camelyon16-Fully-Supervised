import argparse
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
from Dataset import HistologyTileDataset, create_dataloaders 

def parse_args():
    parser = argparse.ArgumentParser(description = 'Tile-level Binary Classifier')

    # Data paths
    parser.add_argument('--pos_slide_path', type=str, required=True,
                        help='Directory containing positive (tumor) WSI slides')
    parser.add_argument('--neg_slide_path', type=str, required=True,
                        help='Directory containing negative (non-tumor) WSI slides')
    parser.add_argument('--pos_grid_path', type=str, required=True,
                        help='Path to pickle file with positive tile coordinates')
    parser.add_argument('--neg_grid_path', type=str, required=True,
                        help='Path to pickle file with negative tile coordinates')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Size of tiles to extract (default: 256)')
    parser.add_argument('--samples_per_class', type=int, default=2000,
                        help='Number of samples to use per class (default: 2000)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs to train (default: 25)')
    parser.add_argument('--validate_every', type=int, default=5,
                        help='Validate model every N epochs (default: 5)')

    # Output parameters
    parser.add_argument('--save_dir', type=str, default='training_history',
                        help='Directory to save models (default: training_history)')
    
    return parser.parse_args()


def set_seed(seed = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    precision = precision_score(y_true, y_pred, average = 'binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average = 'binary', zero_division=0) 
    f1 = f1_score(y_true, y_pred, average = 'binary', zero_division=0) 
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
        accuracy = (tp + tn) / (tp + fp + fn + tn) 
    except:
        accuracy = 0


    return {

        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class SaveBestModel:
    def __init__(self, best_valid_metric = 0.0, metric_name = 'f1', save_dir = 'models'):
        self.best_valid_metric = best_valid_metric
        self.metric_name = metric_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok = True)

    

    def __call__(self, current_valid_metric, epoch, model, optimizer, metrics):
        if current_valid_metric > self.best_valid_metric:
            self.best_valid_metric = current_valid_metric
            print(f"\nBest validation {self.metric_name}: {self.best_valid_metric:.4f}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(self.save_dir, 'best_model.pth'))


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs = 25, validate_every=5):
    
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
        
        for inputs, labels in train_loader:
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

            with torch.no_grad():
                for inputs, labels in val_loader:
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


def plot_training_history(history, samples_per_class):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'])
    plt.plot(range(0, len(history['val_loss'])*5, 5), history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train', 'Val'])

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot([m[metric] for m in history['train_metrics']])
        plt.plot(range(0, len(history['val_metrics'])*5, 5), [m[metric] for m in history['val_metrics']])
        plt.title(metric)
        plt.legend(['Train', 'Val'])

    plt.tight_layout()
    os.makedirs('training_history', exist_ok=True)
    plt.savefig(f'training_history/training_history{samples_per_class}.png')
    plt.show()   


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok = True)

    train_loader, val_loader = create_dataloaders(
        args.pos_slide_path, args.neg_slide_path,
        args.pos_grid_path, args.neg_grid_path,
        args.batch_size, args.tile_size, args.samples_per_class, args.num_workers
    )

    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    save_best_model = SaveBestModel(metric_name='f1', save_dir='training_history/models')

    model, history = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epochs, args.validate_every)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, 'training_history/models/final_model.pth')

    plot_training_history(history, args.samples_per_class)