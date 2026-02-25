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

# Fix for OpenMP library conflict (common on Windows with PyTorch/Matplotlib)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_data_loaders(data_dir, batch_size=32, train_split=0.8, img_size=(224, 224)):
    """
    Creates DataLoaders for train and validation sets with advanced preprocessing.
    """
    # Use external transformations
    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)

    # Load the datasets
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Split the dataset indices
    num_data = len(full_dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    
    split = int(np.floor(train_split * num_data))
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Create subsets with their respective transforms
    # ImageFolder.transform is applied during __getitem__
    # So we wrap the subsets to override the transform
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)

    # Note: Reset base dataset transform to None to handle it in Subset
    full_dataset.transform = None 
    train_subset = TransformedSubset(Subset(full_dataset, train_indices), transform=train_transforms)
    val_subset = TransformedSubset(Subset(full_dataset, val_indices), transform=val_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, full_dataset.classes

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
