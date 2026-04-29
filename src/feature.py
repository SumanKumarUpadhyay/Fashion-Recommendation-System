import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

_base = models.mobilenet_v2(pretrained=True)
model = torch.nn.Sequential(
    _base.features,
    torch.nn.AdaptiveAvgPool2d((1, 1))  # ← critical fix
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(img):
    try:
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = model(tensor)        # (1, 1280, 1, 1)

        features = features.squeeze().numpy()  # (1280,)
        features = features / np.linalg.norm(features)
        return features

    except Exception as e:
        print(f"[feature.py] Error: {e}")   # check terminal for actual error
        return None