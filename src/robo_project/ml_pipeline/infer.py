import torch
from PIL import Image

from torchvision import transforms

from model import PoseEstimator


device = torch.device("cpu")

model = PoseEstimator().to(device)

model.load_state_dict(
    torch.load("pose_estimator.pth", map_location=device)
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_pose(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        prediction = model(image)

    x = prediction[0][0].item()

    z = prediction[0][1].item()

    return x, z


image_path = "../habitat_dataset/front/000100.png"

x, z = predict_pose(image_path)

print(f"Predicted X: {x:.4f}")

print(f"Predicted Z: {z:.4f}")
