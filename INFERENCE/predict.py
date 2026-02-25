import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MODEL.model import get_model
from DATA_PREPROCESSING.preprocessing import get_val_transforms

def predict_single_image(image_path, model_path='best_model.pth', classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['Excellent', 'Good', 'Fair', 'Poor']

    model = get_model(num_classes=len(classes)).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Error: {model_path} not found.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        transform = get_val_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_class = classes[preds[0].item()]
    confidence = probabilities[0][preds[0].item()].item()

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted Category: {predicted_class}")
    print(f"Confidence Score: {confidence:.4f}")

    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_single_image(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_image>")
        test_img = r"d:\core_ml\dataset\ADVE\0000136188.jpg"
        if os.path.exists(test_img):
            print(f"\n--- Running Demo Prediction on {test_img} ---")
            predict_single_image(test_img)
