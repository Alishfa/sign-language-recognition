import cv2
import mediapipe as mp
import os


# PROJECT PATH


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset_v2")



# MEDIAPIPE


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)



# DATASET FOLDER

def get_dataset_path(label):
    path = os.path.join(DATASET_DIR, label)
    os.makedirs(path, exist_ok=True)
    return path

# COLLECTOR NAME


collector_name = input(  "Enter your name (examle:Aqsa): ").strip().replace(" ", "_")

if not collector_name:
    print("Error: Collector name cannot be empty.")
    exit()


# GET INITIAL LABEL

label = input("Enter alphabet (A-Z): ").strip().upper()

if len(label) != 1 or not label.isalpha() or not label.isupper():
    print("Error: Please enter one alphabet from A-Z.")
    exit()


dataset_path = get_dataset_path(label)



# IMAGE COUNTER
# Count only this collector's images


def get_image_count(label, collector_name):

    folder_path = get_dataset_path(label)

    prefix = f"{label}_{collector_name}_"

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and f.startswith(prefix)
    ]

    return len(files)


img_count = get_image_count(label, collector_name)


print()
print(f"Collector: {collector_name}")
print(f"Collecting images for: {label}")
print(f"Your existing images for {label}: {img_count}")
print()
print("S = Save image")
print("N = Change letter")
print("Q = Quit")


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# MAIN LOOP

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break


    # Mirror effect
   

    frame = cv2.flip(frame, 1)


    

    clean_frame = frame.copy()


    # Convert BGR → RGB for MediaPipe

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hand

    results = hands.process(rgb_frame)

    cropped_hand = None


    # HAND DETECTED

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape


        # Get landmark coordinates
        

        x_list = []
        y_list = []


        for landmark in hand_landmarks.landmark:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            x_list.append(x)
            y_list.append(y)


        # Initial bounding box

        x_min = min(x_list)
        y_min = min(y_list)

        x_max = max(x_list)
        y_max = max(y_list)


        # Calculate hand size

        box_width = x_max - x_min
        box_height = y_max - y_min


        # Proportional padding

        padding = int(
            max(box_width, box_height) * 0.30
        )


        x_min -= padding
        y_min -= padding

        x_max += padding
        y_max += padding


        # Make box square
        

        box_width = x_max - x_min
        box_height = y_max - y_min

        size = max(
            box_width,
            box_height
        )

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2


        x_min = center_x - size // 2
        x_max = center_x + size // 2

        y_min = center_y - size // 2
        y_max = center_y + size // 2


        # Keep coordinates inside frame

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)

        x_max = min(x_max, w)
        y_max = min(y_max, h)


        # Crop CLEAN frame

        cropped_hand = clean_frame[
            y_min:y_max,
            x_min:x_max
        ]


        # Draw box for DISPLAY ONLY

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )


        # Draw landmarks for DISPLAY ONLY

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


    # DISPLAY INFORMATION

    cv2.putText(
        frame,
        f"Collector: {collector_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    cv2.putText(
        frame,
        f"Letter: {label}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )


    cv2.putText(
        frame,
        f"Your Images: {img_count}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "S = Save | N = Next Letter | Q = Quit",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show camera

    cv2.imshow(
        "Sign Language Dataset Collection",
        frame
    )


    # KEYBOARD INPUT

    key = cv2.waitKey(1) & 0xFF

    # SAVE IMAGE

    if key == ord("s"):

        if cropped_hand is not None:

            # Resize clean crop
            save_image = cv2.resize(
                cropped_hand,
                (300, 300)
            )


            filename = os.path.join(
                dataset_path,
                f"{label}_{collector_name}_{img_count}.jpg"
            )


            success = cv2.imwrite(
                filename,
                save_image
            )


            if success:

                print(
                    f"Saved: {filename}"
                )

                img_count += 1

            else:

                print(
                    "Error: Image could not be saved."
                )

        else:

            print(
                "No hand detected. Image not saved."
            )

    # Move to next alphabet

    elif key == ord("n"):
        if label == "Z":
            label = "A"
        else:
            label = chr(ord(label) + 1)

        # Update dataset folder
        dataset_path = get_dataset_path(label)

        # Count this collector's existing images
        img_count = get_image_count(
            label,
            collector_name
        )

        print(f"\nNow collecting: {label}")
        print(f"Your existing images: {img_count}")

    # QUIT

    elif key == ord("q"):

        print(
            "Stopping dataset collection..."
        )

        break


# CLEANUP


cap.release()

cv2.destroyAllWindows()

hands.close()