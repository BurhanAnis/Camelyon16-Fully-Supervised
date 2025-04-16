import argparse
import psutil
from datetime import datetime
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

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # in MB
    return 0.0

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
    parser.add_argument('--batch_size', type=int, default=32,  # Increased default batch size
                        help='Batch size for training (default: 32)')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Size of tiles to extract (default: 256)')
    parser.add_argument('--samples_per_class', type=int, default=2000,
                        help='Number of samples to use per class (default: 2000)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading (default: 8)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches loaded in advance by each worker (default: 2)')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Keep workers alive between iterations')
    parser.add_argument('--cache_size', type=int, default=5,
                        help='Number of slides to keep in memory cache (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs to train (default: 25)')
    parser.add_argument('--validate_every', type=int, default=5,
                        help='Validate model every N epochs (default: 5)')
    parser.add_argument('--use_amp', action='store_true', 
                        help='Use automatic mixed precision training')

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


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics efficiently with minimal tensor conversions
    
    Args:
        y_true: Tensor of ground truth labels
        y_pred: Tensor of predicted labels
        
    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    # Only convert to numpy once
    if torch.is_tensor(y_true):
        y_true_np = y_true.cpu().numpy()
    else:
        y_true_np = y_true
        
    if torch.is_tensor(y_pred):
        y_pred_np = y_pred.cpu().numpy()
    else:
        y_pred_np = y_pred

    # Calculate metrics more efficiently
    precision = precision_score(y_true_np, y_pred_np, average='binary', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='binary', zero_division=0) 
    f1 = f1_score(y_true_np, y_pred_np, average='binary', zero_division=0) 
    
    # Calculate confusion matrix only once
    try:
        cm = confusion_matrix(y_true_np, y_pred_np)
        tn, fp, fn, tp = cm.ravel() 
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


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=25, validate_every=5, save_dir='training_history', device = None):
    """Optimized training loop that reduces tensor operations and eliminates bottlenecks"""
    log_file = os.path.join(save_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {train_loader.batch_size}, Num workers: {train_loader.num_workers}\n")
        f.write("-" * 80 + "\n")

    total_start_time = time.time()
    
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

    # Pre-allocate tensors for labels and predictions to avoid repeated allocations
    all_train_labels = torch.zeros(len(train_loader.dataset), dtype=torch.long)
    all_train_preds = torch.zeros(len(train_loader.dataset), dtype=torch.long)
    
    if validate_every > 0:
        all_val_labels = torch.zeros(len(val_loader.dataset), dtype=torch.long)
        all_val_preds = torch.zeros(len(val_loader.dataset), dtype=torch.long)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        epoch_start_time = time.time()
        mem_start = psutil.virtual_memory().used / (1024 ** 3)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_start = torch.cuda.memory_allocated() / 1024**2
        else:
            gpu_start = 0.0

        model.train()
        running_loss = 0.0
        training_start_idx = 0
        
        # Use automatic mixed precision for faster training if available
        scaler = torch.cuda.amp.GradScaler() if hasattr(torch.cuda, 'amp') and device.type == 'cuda' else None
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
            
            if scaler is not None:
                # Use mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Compute predictions without using .item() in the hot loop
            _, preds = torch.max(outputs, 1)
            
            # Store batch statistics
            running_loss += loss.item() * batch_size
            
            # Store labels and predictions directly without converting to numpy
            end_idx = training_start_idx + batch_size
            all_train_labels[training_start_idx:end_idx] = labels.cpu()
            all_train_preds[training_start_idx:end_idx] = preds.cpu()
            training_start_idx = end_idx
            
        epoch_end_time = time.time()
        mem_end = psutil.virtual_memory().used / (1024 ** 3)
        cpu_percent = psutil.cpu_percent(interval=None)
        gpu_end = get_gpu_memory()
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            gpu_peak = 0.0

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Only calculate metrics once per epoch, not per batch
        train_metrics = calculate_metrics(
            all_train_labels[:training_start_idx], 
            all_train_preds[:training_start_idx]
        )
        
        history['train_loss'].append(epoch_loss)
        history['train_metrics'].append(train_metrics)

        print(f'Train Loss: {epoch_loss:.4f}')
        for k, v in train_metrics.items():
            print(f'  {k}: {v:.4f}')

        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
            f.write(f"  Time: {epoch_end_time - epoch_start_time:.2f}s\n")
            f.write(f"  CPU usage: {cpu_percent:.2f}%\n")
            f.write(f"  Memory used: {mem_end - mem_start:.2f} GB\n")
            f.write(f"  GPU memory used: {gpu_end - gpu_start:.2f} MB\n")
            f.write(f"  Peak GPU memory used: {gpu_peak:.2f} MB\n")

        # Validation - only do it periodically to save time
        if validate_every > 0 and ((epoch + 1) % validate_every == 0 or epoch == num_epochs - 1):
            model.eval()
            running_loss = 0.0
            val_start_idx = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    batch_size = inputs.size(0)
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    running_loss += loss.item() * batch_size
                    
                    # Store batch results
                    end_idx = val_start_idx + batch_size
                    all_val_labels[val_start_idx:end_idx] = labels.cpu()
                    all_val_preds[val_start_idx:end_idx] = preds.cpu()
                    val_start_idx = end_idx
                    
            val_loss = running_loss / len(val_loader.dataset)
            
            val_metrics = calculate_metrics(
                all_val_labels[:val_start_idx],
                all_val_preds[:val_start_idx]
            )
            
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

    total_time = time.time() - total_start_time
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now()}\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")

    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history, samples_per_class, save_dir='training_history'):
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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'training_history_s{samples_per_class}.png'))
    plt.show()   


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok = True)

    set_seed()

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        args.pos_slide_path, args.neg_slide_path,
        args.pos_grid_path, args.neg_grid_path,
        args.batch_size, args.tile_size, args.samples_per_class, 
        args.num_workers, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers
    )


    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    if args.use_amp and device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        print("Using Automatic Mixed Precision (AMP) training")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    save_best_model = SaveBestModel(metric_name='f1', save_dir=os.path.join(args.save_dir, 'models'))

    model, history = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        args.num_epochs, 
        args.validate_every, 
        save_dir = args.save_dir,
        device = device
        )

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(args.save_dir, 'models', 'final_model.pth'))
    

    plot_training_history(history, args.samples_per_class, save_dir=args.save_dir)