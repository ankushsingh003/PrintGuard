import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import random

def apply_quality_level(image, level):
    if level == 0:
        return image
    if level == 1:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        return _add_gaussian_noise(image, intensity=5)
    elif level == 2:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 1.8)))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.6, 0.8))
        return _add_gaussian_noise(image, intensity=15)
    elif level == 3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 4.0)))
        width, height = image.size
        small = image.resize((width//4, height//4), resample=Image.BILINEAR)
        image = small.resize((width, height), resample=Image.NEAREST)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.4, 0.6))
        return _add_gaussian_noise(image, intensity=30)
    return image

def _add_gaussian_noise(image, intensity=10):
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, intensity, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

if __name__ == "__main__":
    test_img_path = r"d:\core_ml\dataset\ADVE\0000136188.jpg"
    try:
        img = Image.open(test_img_path)
        for i in range(4):
            degraded = apply_quality_level(img, i)
            degraded.save(f"quality_level_{i}.jpg")
        print("Test degradations saved as quality_level_0/1/2/3.jpg")
    except Exception as e:
        print(f"Test failed: {e}")
