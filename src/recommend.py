import numpy as np

def recommend(features, feature_list, top_k=3):
    try:
        if features is None or len(feature_list) == 0:
            return [], []
        features = np.array(features, dtype=np.float32)
        feature_list = np.array(feature_list, dtype=np.float32)
        norms = np.linalg.norm(feature_list, axis=1, keepdims=True)
        feature_list = feature_list / (norms + 1e-10)
        similarities = np.dot(feature_list, features)
        indices = np.argsort(similarities)[-top_k:][::-1]
        scores = similarities[indices]
        return indices.tolist(), scores.tolist()
    except Exception as e:
        print("Recommend error:", e)
        return [], []
