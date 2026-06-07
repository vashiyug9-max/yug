import os
import torch

from PIL import Image

from torchvision import transforms

from robo_project.model import PoseEstimator


device = torch.device("cpu")


model = PoseEstimator().to(device)


model_path = os.path.expanduser(
    "~/robo_project_ws/src/robo_project/robo_project/pose_estimator.pth"
)


model.load_state_dict(
    torch.load(model_path, map_location=device)
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
