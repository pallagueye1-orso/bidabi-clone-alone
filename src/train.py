import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data/raw/images"

X = []
y = []

# -------------------------
# LOAD DATA + AUGMENTATION
# -------------------------
for i, category in enumerate(os.listdir(DATA_DIR)):
    category_path = os.path.join(DATA_DIR, category)

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize((64, 64))
            img_array = np.array(img)

            # ✅ AUGMENTATION (flip horizontal)
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)

            X.append(img_array.flatten())
            y.append(i)
        except:
            continue

X = np.array(X)
y = np.array(y)

# -------------------------
# SPLIT (train / val / test)
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# -------------------------
# MODEL
# -------------------------
model = LogisticRegression(max_iter=1000)

# -------------------------
# TRAIN
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# VALIDATION
# -------------------------
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

print(f"Validation Accuracy: {val_acc:.4f}")

# -------------------------
# TEST
# -------------------------
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print(f"Test Accuracy: {test_acc:.4f}")

# -------------------------
# SAVE BEST MODEL
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

print("✔ Best model saved")
