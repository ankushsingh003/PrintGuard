import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add parent directory to path to allow imports from sibling folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DATA_PREPROCESSING.preprocessing import get_train_transforms, get_val_transforms
from DATA_PREPROCESSING.quality_degradation import apply_quality_level

# Fix for OpenMP library conflict (common on Windows with PyTorch/Matplotlib)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class QualityDataset(torch.utils.data.Dataset):
    """
    Wraps a standard image dataset and applies synthetic quality degradation labels.
    Labels: 0 (Excellent), 1 (Good), 2 (Fair), 3 (Poor)
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.quality_levels = [0, 1, 2, 3]

    def __len__(self):
        # Sample each base image at each quality level to maximize data usage
        return len(self.base_dataset) * len(self.quality_levels)

    def __getitem__(self, idx):
        base_idx = idx // len(self.quality_levels)
        quality_level = idx % len(self.quality_levels)
        
        # Get base image (ignoring original label)
        image, _ = self.base_dataset[base_idx]
        
        # Apply synthetic degradation
        degraded_image = apply_quality_level(image, quality_level)
        
        if self.transform:
            degraded_image = self.transform(degraded_image)
            
        return degraded_image, torch.tensor(quality_level)

def get_data_loaders(data_dir, batch_size=32, train_split=0.8, img_size=(224, 224), subset_fraction=1.0):
    """
    Creates DataLoaders for Print Quality Detection. 
    Labels are quality levels [0-3] instead of document categories.
    """
    # Use standard transformations
    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)

    # Load the base dataset (ImageFolder)
    base_dataset = datasets.ImageFolder(root=data_dir)
    
    # Optional: Take a subset of the base data to speed up training
    if subset_fraction < 1.0:
        num_subset = int(len(base_dataset) * subset_fraction)
        indices = list(range(len(base_dataset)))
        np.random.seed(42)
        np.random.shuffle(indices)
        subset_indices = indices[:num_subset]
        base_dataset = Subset(base_dataset, subset_indices)

    # Split the base dataset
    num_data = len(base_dataset)
    indices = list(range(num_data))
    np.random.seed(42) # Set seed for reproducible split
    np.random.shuffle(indices)
    
    split = int(np.floor(train_split * num_data))
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Wrap subsets with QualityDataset
    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)
    
    train_dataset = QualityDataset(train_subset, transform=train_transforms)
    val_dataset = QualityDataset(val_subset, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    quality_classes = ['Excellent', 'Good', 'Fair', 'Poor']
    
    return train_loader, val_loader, quality_classes

def visualize_batch(images, labels, classes):
    """
    Shows a batch of images from tensors.
    """
    try:
        plt.figure(figsize=(12, 8))
        num_images = min(8, len(images))
        for i in range(num_images):
            plt.subplot(2, 4, i + 1)
            # Rescale image from [-1, 1] to [0, 1] for visualization
            img = images[i].numpy().transpose((1, 2, 0))
            img = 0.5 * img + 0.5
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(classes[labels[i]])
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        plt.close() # Always close to free up backend resources

if __name__ == "__main__":
    DATA_DIR = r"d:\core_ml\dataset"
    
    if os.path.exists(DATA_DIR):
        print(f"--- Loading data from {DATA_DIR} ---")
        train_loader, val_loader, classes = get_data_loaders(DATA_DIR)
        
        print(f"Number of classes: {len(classes)}")
        print(f"Classes: {classes}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Test a single batch with a guard
        try:
            images, labels = next(iter(train_loader))
            print(f"\nBatch Information:")
            print(f"Image tensor shape: {images.shape}")
            print(f"Label tensor shape: {labels.shape}")
            print(f"Unique classes in this batch: {torch.unique(labels).tolist()}")
            
            # Pass images/labels directly to the visualization function
            visualize_batch(images, labels, classes)
            print("\n--- Pipeline successfully verified ---")
        except Exception as e:
            print(f"\nError during batch testing: {e}")
            print("This could be due to a corrupted image file in the dataset.")
    else:
        print(f"Error: Directory {DATA_DIR} not found.")
