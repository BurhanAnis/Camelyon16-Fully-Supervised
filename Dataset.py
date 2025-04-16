import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import pickle
import random
import openslide
import torch 
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time

class HistologyTileDataset(Dataset):
    def __init__(self, pos_slide_path, neg_slide_path, pos_grid_path, neg_grid_path, 
                 tile_size=256, transform=None, samples_per_class=5000, cache_size = 5):
        """
        Args:
            pos_slide_path (str): Directory containing positive (tumor) WSI slides
            neg_slide_path (str): Directory containing negative (non-tumor) WSI slides
            pos_grid_path (str): Path to pickle file with positive tile coordinates
            neg_grid_path (str): Path to pickle file with negative tile coordinates
            tile_size (int): Size of tiles to extract (assuming square tiles)
            transform: PyTorch transforms for data augmentation
            samples_per_class (int): Number of samples to use per class
        """
        self.pos_slide_path = pos_slide_path
        self.neg_slide_path = neg_slide_path
        self.tile_size = tile_size
        self.transform = transform
        self.cache_size = cache_size
        
        # Load coordinate grids
        with open(pos_grid_path, 'rb') as f:
            self.pos_grid = pickle.load(f)
        
        with open(neg_grid_path, 'rb') as f:
            self.neg_grid = pickle.load(f)
        
        # Get slide names
        self.pos_slides = [f for f in os.listdir(pos_slide_path) if f.endswith('.tif')]
        self.neg_slides = [f for f in os.listdir(neg_slide_path) if f.endswith('.tif')]
        self.pos_slide_full_paths = [os.path.join(pos_slide_path, slide) for slide in self.pos_slides]
        self.neg_slide_full_paths = [os.path.join(neg_slide_path, slide) for slide in self.neg_slides]

        
        # Prepare sample indices
        self.pos_samples = self._prepare_samples(self.pos_grid, 1, samples_per_class)
        self.neg_samples = self._prepare_samples(self.neg_grid, 0, samples_per_class)

        # Group samples by slide to improve cache locality
        self.samples = self._group_by_slide(self.pos_samples + self.neg_samples)
        
        # LRU cache for slide objects to improve performance
        self.slide_cache = {}
        self.slide_cache_usage = {}  # Track last usage time for each slid

    def _prepare_samples(self, grid, label, num_samples):
        """Prepare samples from a grid with associated label"""
        total_samples = sum(len(tiles) for tiles in grid)  # total number of available tile coordinates
        samples = []

        if total_samples <= num_samples:
            # Use all samples and sample with replacement if needed
            for slide_idx, tiles in enumerate(grid):
                for tile_idx in range(len(tiles)):
                    samples.append({
                        'slide_idx': slide_idx,
                        'tile_idx': tile_idx,
                        'label': label
                    })

            # If we need more, sample with replacement
            if len(samples) < num_samples:
                extra_samples = random.choices(samples, k=num_samples - len(samples))
                samples.extend(extra_samples)

        else:
            # Limit memory: sample a random subset of tile locations
            fraction = min(1.0, 2.0 * num_samples / total_samples)

            for slide_idx, tiles in enumerate(grid):
                for tile_idx in range(len(tiles)):
                    if random.random() <= fraction:
                        samples.append({
                            'slide_idx': slide_idx,
                            'tile_idx': tile_idx,
                            'label': label
                        })

            # Ensure we return exactly num_samples
            if len(samples) > num_samples:
                samples = random.sample(samples, num_samples)
            elif len(samples) < num_samples:
                extra_samples = random.choices(samples, k=num_samples - len(samples))
                samples.extend(extra_samples)

        return samples


    def _group_by_slide(self, samples):
            """Group samples by slide to improve cache hit rate"""
            # Sort samples by slide_idx to improve cache locality
            samples.sort(key=lambda x: (x['label'], x['slide_idx']))
            return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        slide_idx = sample['slide_idx']
        tile_idx = sample['tile_idx']
        
        # Determine which set of slides and grid to use
        if label == 1:  # Positive/tumor
            slide_path = self.pos_slide_full_paths[slide_idx]
            grid = self.pos_grid
        else:  # Negative/non-tumor
            slide_path = self.neg_slide_full_paths[slide_idx]
            grid = self.neg_grid

        # Get tile coordinates
        x, y = grid[slide_idx][tile_idx]
        
        # Load slide (use LRU cache for efficiency)
        slide = self._get_slide(slide_path)


        # Extract tile
        tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
        tile = tile.convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            tile = self.transform(tile)
        else:
            # Convert PIL image to tensor
            # tile = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float() / 255.0
            raise ValueError("Transform not provided — use transforms.ToTensor() at minimum.")
        
        return tile, label

    def _get_slide(self, slide_path):
        """Get slide from cache with LRU replacement policy"""
        # Update usage time for this slide
        current_time = time.time()
        
        if slide_path in self.slide_cache:
            # Update last access time
            self.slide_cache_usage[slide_path] = current_time
            return self.slide_cache[slide_path]
        
        # Load new slide
        slide = openslide.OpenSlide(slide_path)
        self.slide_cache[slide_path] = slide
        self.slide_cache_usage[slide_path] = current_time
        
        # Check if cache is full
        if len(self.slide_cache) > self.cache_size:
            # Find least recently used slide
            lru_slide = min(self.slide_cache_usage.items(), key=lambda x: x[1])[0]
            # Remove it from cache
            del self.slide_cache[lru_slide]
            del self.slide_cache_usage[lru_slide]
            
        return slide

def create_dataloaders(pos_slide_path, neg_slide_path, pos_grid_path, neg_grid_path, 
                       batch_size=32, tile_size=256, samples_per_class=5000, 
                       num_workers=4, val_split=0.2, prefetch_factor=2, persistent_workers = True):
    """
    Create training and validation dataloaders
    
    Args:
        pos_slide_path (str): Directory containing positive WSI slides
        neg_slide_path (str): Directory containing negative WSI slides
        pos_grid_path (str): Path to pickle file with positive tile coordinates
        neg_grid_path (str): Path to pickle file with negative tile coordinates
        batch_size (int): Batch size for training
        tile_size (int): Size of tiles to extract
        samples_per_class (int): Number of samples to use per class
        num_workers (int): Number of worker processes for data loading
        val_split (float): Fraction of data to use for validation
        prefetch_factor (int): Number of batches loaded in advance by each worker
        persistent_workers (bool): Whether to keep worker processes alive between iterations
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Calculate samples for train and validation
    train_samples = int(samples_per_class * (1 - val_split))
    val_samples = int(samples_per_class * val_split)
    
    # Create datasets with optimized caching
    train_dataset = HistologyTileDataset(
        pos_slide_path=pos_slide_path,
        neg_slide_path=neg_slide_path,
        pos_grid_path=pos_grid_path,
        neg_grid_path=neg_grid_path,
        tile_size=tile_size,
        transform=train_transform,
        samples_per_class=train_samples,
        cache_size=min(10, num_workers * 2)  # Optimize cache size based on workers
    )
    
    val_dataset = HistologyTileDataset(
        pos_slide_path=pos_slide_path,
        neg_slide_path=neg_slide_path,
        pos_grid_path=pos_grid_path,
        neg_grid_path=neg_grid_path,
        tile_size=tile_size,
        transform=val_transform,
        samples_per_class=val_samples,
        cache_size=min(10, num_workers * 2)  # Optimize cache size based on workers
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader     