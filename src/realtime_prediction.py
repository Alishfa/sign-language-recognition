import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "sign_language_model.keras")
model = load_model(model_path)
labels = {
    0:"A",
    1:"B",
    2:"C"
}
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break
    frame = cv2.flip(frame, 1)
    cv2.rectangle(
         frame,
         (300,100),
        (600,400),
        (0,255,0),
        2
    )
    roi=frame[100:400,300:600]
    img=cv2.resize(roi,(64,64))
    img=img/255.0
    img = np.expand_dims(img,axis=0)
    prediction = model.predict(img,verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    letter=labels[predicted_class]
    cv2.putText(
        frame,
        f"prediction: {letter}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )
    cv2.imshow("Sign Language Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

cap.release()
cv2.destroyAllWindows()