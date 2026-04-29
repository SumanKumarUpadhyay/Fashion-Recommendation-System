import os
import pickle
from PIL import Image
from src.feature import extract_features

IMAGE_FOLDER = "Data/images"
VALID_EXT = ('.jpg', '.jpeg', '.png', '.webp')

features, filenames = [], []

print("Generating embeddings...")

for file in os.listdir(IMAGE_FOLDER):
    if not file.lower().endswith(VALID_EXT):
        continue
    path = os.path.join(IMAGE_FOLDER, file)
    try:
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            filenames.append(path)
            print(f"✅ {file}")
    except Exception as e:
        print(f"❌ {file}: {e}")

os.makedirs("models", exist_ok=True)
pickle.dump(features, open("models/image_features.pkl", "wb"))
pickle.dump(filenames, open("models/filenames.pkl", "wb"))
print(f"\n✅ Saved {len(features)} embeddings.")