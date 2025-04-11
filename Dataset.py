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

class HistologyTileDataset(Dataset):
    def __init__(self, pos_slide_path, neg_slide_path, pos_grid_path, neg_grid_path, 
                 tile_size=256, transform=None, samples_per_class=5000):
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
        
        # Load coordinate grids
        with open(pos_grid_path, 'rb') as f:
            self.pos_grid = pickle.load(f)
        
        with open(neg_grid_path, 'rb') as f:
            self.neg_grid = pickle.load(f)
        
        # Get slide names
        self.pos_slides = [f for f in os.listdir(pos_slide_path) if f.endswith('.tif')]
        self.neg_slides = [f for f in os.listdir(neg_slide_path) if f.endswith('.tif')]
        
        # Prepare sample indices
        self.pos_samples = self._prepare_samples(self.pos_grid, 1, samples_per_class)
        self.neg_samples = self._prepare_samples(self.neg_grid, 0, samples_per_class)
        
        # Combine samples
        self.samples = self.pos_samples + self.neg_samples
        random.shuffle(self.samples)
        
        # Cache for open slides to improve performance
        self.slide_cache = {}

    def _prepare_samples(self, grid, label, num_samples):
        """Prepare samples from a grid with associated label"""
        samples = []
        all_possible_samples = []
        
        # Create list of all possible (slide_idx, tile_idx) combinations
        for slide_idx, tiles in enumerate(grid):
            for tile_idx in range(len(tiles)):
                all_possible_samples.append({
                    'slide_idx': slide_idx,
                    'tile_idx': tile_idx,
                    'label': label
                })
        
        # Randomly select num_samples
        if len(all_possible_samples) > num_samples:
            samples = random.sample(all_possible_samples, num_samples)
        else:
            # If we don't have enough samples, use all and sample with replacement
            samples = random.choices(all_possible_samples, k=num_samples)
        
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
            slide_path = self.pos_slide_path
            slides = self.pos_slides
            grid = self.pos_grid
        else:  # Negative/non-tumor
            slide_path = self.neg_slide_path
            slides = self.neg_slides
            grid = self.neg_grid
        
        # Get slide filename
        slide_filename = slides[slide_idx]
        full_slide_path = os.path.join(slide_path, slide_filename)
        
        # Get tile coordinates
        x, y = grid[slide_idx][tile_idx]
        
        # Load slide (use cache for efficiency)
        if full_slide_path not in self.slide_cache:
            self.slide_cache[full_slide_path] = openslide.OpenSlide(full_slide_path)
            
            # Implement a simple cache size limit
            if len(self.slide_cache) > 10:  # Keep only 10 slides in memory
                # Remove a random slide from cache
                remove_key = random.choice(list(self.slide_cache.keys()))
                if remove_key != full_slide_path:
                    del self.slide_cache[remove_key]
                    
        slide = self.slide_cache[full_slide_path]
        
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

def create_dataloaders(pos_slide_path, neg_slide_path, pos_grid_path, neg_grid_path, 
                       batch_size=32, tile_size=256, samples_per_class=5000, 
                       num_workers=4, val_split=0.2):
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
    
    # Create datasets
    train_dataset = HistologyTileDataset(
        pos_slide_path=pos_slide_path,
        neg_slide_path=neg_slide_path,
        pos_grid_path=pos_grid_path,
        neg_grid_path=neg_grid_path,
        tile_size=tile_size,
        transform=train_transform,
        samples_per_class=train_samples
    )
    
    val_dataset = HistologyTileDataset(
        pos_slide_path=pos_slide_path,
        neg_slide_path=neg_slide_path,
        pos_grid_path=pos_grid_path,
        neg_grid_path=neg_grid_path,
        tile_size=tile_size,
        transform=val_transform,
        samples_per_class=val_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader    