import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import random

def apply_quality_level(image, level):
    """
    Applies a specific quality level to a PIL image.
    0: Excellent (Original)
    1: Good (Minor noise/blur)
    2: Fair (Significant noise, blur, lower contrast)
    3: Poor (Severe degradation)
    """
    if level == 0:
        return image
    
    # Levels are cumulative or specific based on severity
    if level == 1:
        # Good Quality: Add very slight blur and noise
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        return _add_gaussian_noise(image, intensity=5)
        
    elif level == 2:
        # Fair Quality: Noticeable blur and noise, lower contrast
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 1.8)))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.6, 0.8))
        return _add_gaussian_noise(image, intensity=15)
        
    elif level == 3:
        # Poor Quality: Severe blur, pixelation, and heavy noise
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 4.0)))
        
        # Add pixelation
        width, height = image.size
        small = image.resize((width//4, height//4), resample=Image.BILINEAR)
        image = small.resize((width, height), resample=Image.NEAREST)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.4, 0.6))
        
        return _add_gaussian_noise(image, intensity=30)
    
    return image

def _add_gaussian_noise(image, intensity=10):
    """Helper to add Gaussian noise to a PIL image."""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, intensity, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

if __name__ == "__main__":
    # Test degradation
    test_img_path = r"d:\core_ml\dataset\ADVE\0000136188.jpg"
    try:
        img = Image.open(test_img_path)
        for i in range(4):
            degraded = apply_quality_level(img, i)
            degraded.save(f"quality_level_{i}.jpg")
        print("Test degradations saved as quality_level_0/1/2/3.jpg")
    except Exception as e:
        print(f"Test failed: {e}")
