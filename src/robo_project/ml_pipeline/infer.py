import os
import torch
from PIL import Image
from torchvision import transforms
from ament_index_python.packages import get_package_share_directory

# Absolute namespace import for the model structure
from robo_project.ml_pipeline.model import PoseEstimator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX: Initialize the model instance so load_state_dict has an object to populate
model = PoseEstimator()

# FIX: Leverage ament tracking to locate your model files inside the share directory
try:
    SHARE_DIR = get_package_share_directory('robo_project')
    WEIGHTS_PATH = os.path.join(SHARE_DIR, "ml_pipeline", "pose_estimator.pth")
except Exception:
    # Fallback backup for external testing out of a sourced terminal env
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(CURRENT_DIR, "pose_estimator.pth")

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Model weights not found at: {WEIGHTS_PATH}")

# Load the verified weight maps and pass the model to the target device
model_state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(model_state)
model.to(device)
model.eval()

# Image transforms pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_pose(image_path):
    # FIX: Expand home tildes (~) so PIL doesn't drop a FileNotFoundError
    expanded_path = os.path.abspath(os.path.expanduser(image_path))
    
    image = Image.open(expanded_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        prediction = model(image)

    x = prediction[0][0].item()
    z = prediction[0][1].item()
    return x, z

def predict_pose_from_pil(image):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        prediction = model(image)

    x = prediction[0][0].item()
    z = prediction[0][1].item()
    return x, z

# Test sequence tracking execution block
if __name__ == "__main__":
    image_path = "~/robo_project_ws/src/dataset/00100_front.jpg"
    try:
        x, z = predict_pose(image_path)
        print(f"Predicted X: {x:.4f}")
        print(f"Predicted Z: {z:.4f}")
    except FileNotFoundError as e:
        print(f"Skipping prediction test block: {e}")
