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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
set_seed(42)

# Function to determine available device(s)
def get_device_setup():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Found {num_gpus} GPUs! Will use DistributedDataParallel.")
            use_ddp = True
        else:
            print(f"Found 1 GPU. Will use single GPU training.")
            use_ddp = False
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS device.")
        device = torch.device("mps")
        use_ddp = False
    else:
        print("Using CPU.")
        device = torch.device("cpu")
        use_ddp = False
    
    return device, use_ddp, torch.cuda.device_count() if torch.cuda.is_available() else 0


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


class SaveBestModel:
    """
    Class to save the best model during training
    """
    def __init__(self, metric_name='f1', save_dir='histology_models'):
        self.best_metric = 0.0
        self.metric_name = metric_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(self, current_metric, epoch, model, optimizer, metrics):
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            print(f"Saving best model with {self.metric_name}: {current_metric:.4f}")
            
            # Save only on main process if using DDP
            if not hasattr(model, 'module') or dist.get_rank() == 0:
                # If using DDP, need to save the model.module.state_dict()
                model_to_save = model.module if hasattr(model, 'module') else model
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, f'{self.save_dir}/best_model.pth')


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, 
                num_epochs=25, validate_every=5):
    since = time.time()
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    is_distributed = hasattr(model, 'module')
    is_main_process = not is_distributed or (dist.get_rank() == 0)

    if is_main_process:
        print(f"Training on device: {device}")
        print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("-" * 50)

    for epoch in range(num_epochs):
        if is_main_process:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

        # Set train mode
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        # Set epoch for distributed sampler if used
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_pbar = tqdm(train_loader, desc="Training") if is_main_process else train_loader
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
        
        if is_main_process:
            history['train_loss'].append(epoch_loss)
            history['train_metrics'].append(train_metrics)
            print(f'Train Loss: {epoch_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  {k}: {v:.4f}')

        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            model.eval()
            running_loss, all_labels, all_preds = 0.0, [], []
            val_pbar = tqdm(val_loader, desc="Validating") if is_main_process else val_loader

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
            
            if is_main_process:
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                print(f'Validation Loss: {val_loss:.4f}')
                for k, v in val_metrics.items():
                    print(f'  {k}: {v:.4f}')

                # Save best model
                save_best_model(val_metrics['f1'], epoch, model, optimizer, val_metrics)

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_model_wts = model.state_dict()

        scheduler.step()

    time_elapsed = time.time() - since
    if is_main_process:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val F1: {best_f1:.4f}')

    # Load best model weights
    if hasattr(model, 'module'):
        model.module.load_state_dict(best_model_wts)
    else:
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


def setup_distributed(rank, world_size):
    """
    Initialize distributed training process group
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the GPU to use


def cleanup_distributed():
    """
    Clean up distributed process group
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(rank, world_size, args):
    """
    Function to run on each process for distributed training
    """
    # Initialize distributed process group
    if world_size > 1:
        setup_distributed(rank, world_size)
        print(f"Running DDP on rank {rank}.")
    
    # Set device
    device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda")
    
    # Create dataloaders with distributed sampler if using DDP
    if world_size > 1:
        train_dataset = HistologyTileDataset(
            pos_slide_path=args.pos_slide_path,
            neg_slide_path=args.neg_slide_path,
            pos_grid_path=args.pos_grid_path,
            neg_grid_path=args.neg_grid_path,
            tile_size=args.tile_size,
            transform=args.train_transform,
            samples_per_class=args.train_samples
        )
        
        val_dataset = HistologyTileDataset(
            pos_slide_path=args.pos_slide_path,
            neg_slide_path=args.neg_slide_path,
            pos_grid_path=args.pos_grid_path,
            neg_grid_path=args.neg_grid_path,
            tile_size=args.tile_size,
            transform=args.val_transform,
            samples_per_class=args.val_samples
        )
        
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Use the existing create_dataloaders function for single GPU
        train_loader, val_loader = create_dataloaders(
            args.pos_slide_path, args.neg_slide_path,
            args.pos_grid_path, args.neg_grid_path,
            batch_size=args.batch_size, tile_size=args.tile_size, 
            samples_per_class=args.samples_per_class, 
            num_workers=args.num_workers
        )
    
    # Create model
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Initialize SaveBestModel class
    global save_best_model
    save_best_model = SaveBestModel(metric_name='f1', save_dir=args.save_dir)
    
    # Train the model
    model, history = train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, device,
        num_epochs=args.num_epochs, validate_every=args.validate_every
    )
    
    # Save final model (only on main process)
    if rank == 0 or world_size == 1:
        # Save the model
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, f'{args.save_dir}/final_model.pth')
        
        # Plot training history
        plot_training_history(history)
    
    # Clean up distributed process group
    if world_size > 1:
        cleanup_distributed()


def main():
    import argparse
    import torchvision.transforms as transforms
    
    parser = argparse.ArgumentParser(description='Train histology classification model')
    
    # Dataset paths
    parser.add_argument('--pos_slide_path', type=str, 
                        default='/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/positive/images')
    parser.add_argument('--neg_slide_path', type=str, 
                        default='/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/negative')
    parser.add_argument('--pos_grid_path', type=str, 
                        default='/Users/burhananis/fully-supervised-camelyon/data/tumour_grid.pkl')
    parser.add_argument('--neg_grid_path', type=str, 
                        default='/Users/burhananis/fully-supervised-camelyon/data/tile_coords_neg.pkl')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--samples_per_class', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='histology_models')
    
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    args = parser.parse_args()
    
    # Define transformations
    args.train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    args.val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Calculate samples for train and validation
    val_split = 0.2
    args.train_samples = int(args.samples_per_class * (1 - val_split))
    args.val_samples = int(args.samples_per_class * val_split)
    
    # Check if we should use distributed training
    device, use_ddp, num_gpus = get_device_setup()
    
    # If user explicitly asked for distributed training or we detected multiple GPUs
    if args.distributed or (use_ddp and num_gpus > 1):
        import torch.multiprocessing as mp
        
        # Only run distributed training if we have multiple GPUs
        if num_gpus > 1:
            print(f"Starting distributed training on {num_gpus} GPUs")
            # Use all available GPUs
            mp.spawn(
                run_training,
                args=(num_gpus, args),
                nprocs=num_gpus,
                join=True
            )
        else:
            print("Distributed training requested but only one GPU available. Using single GPU training.")
            run_training(0, 1, args)
    else:
        # Run on a single device
        run_training(0, 1, args)


if __name__ == '__main__':
    main()