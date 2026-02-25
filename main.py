import os
import sys

# Entry point for the Core ML Document Classification System
def main():
    print("Document Image Classification System")
    print("="*40)
    print("Modules Available:")
    print("1. DATA_LOADER")
    print("2. DATA_PREPROCESSING")
    print("3. MODEL")
    print("4. TRAINING")
    print("5. EVALUATION")
    print("6. INFERENCE")
    print("\nTo start training, run: python TRAINING/train.py")
    print("To evaluate, run: python EVALUATION/evaluate.py")
    print("To predict, run: python INFERENCE/predict.py <image_path>")

if __name__ == "__main__":
    main()
