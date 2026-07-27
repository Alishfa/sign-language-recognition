import os

from imageprocessing import load_data
from build_model import create_model


# Project Directories


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)



# Training Parameters


EPOCHS = 20
BATCH_SIZE = 16



# Load Dataset


x_train, x_test, y_train, y_test = load_data()



# Number of Classes


num_classes = len(set(y_train))

print(f"Number of Classes: {num_classes}")


# Build CNN Model


model = create_model(num_classes)

model.summary()



# Compile Model


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train Model


history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)



# Evaluate Model


test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")



# Save Model


model_path = os.path.join(
    MODEL_DIR,
    "sign_language_model.keras"
)

model.save(model_path)

print(f"\nModel saved successfully!")
print(model_path)