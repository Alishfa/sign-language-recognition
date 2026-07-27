import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "sign_language_model.keras")
model = load_model(model_path)
labels = {
    0: "A",
    1: "B",
    2: "C"
}
image_path = os.path.join(BASE_DIR, "test_images","C_0.jpg")
image=cv2.imread(image_path)
if image is None:
    print("Error: Unable to read the image.")
    exit()
image = cv2.resize(image, (64, 64))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image,verbose=0)
predicted_index = np.argmax(prediction)
predicted_letter = labels[predicted_index]
confidence = np.max(prediction) * 100
print("\nPrediction Probabilities:")

for i, probability in enumerate(prediction[0]):
    print(f"{labels[i]} : {probability*100:.2f}%")

print("\nFinal Prediction")
print(f"Letter     : {predicted_letter}")
print(f"Confidence : {confidence:.2f}%")
