from torchvision import transforms

def get_train_transforms(img_size=(224, 224)):
    """
    Returns transformations for training with augmentation.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(5), # Simulate slight scan tilt
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Handle slight misalignment
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_val_transforms(img_size=(224, 224)):
    """
    Returns standard transformations for validation.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
