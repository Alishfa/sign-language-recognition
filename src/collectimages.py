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


# GET LABEL

label = input("Enter alphabet (A-Z): ").upper()

dataset_path = get_dataset_path(label)

img_count = len([
    f for f in os.listdir(dataset_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])


print()
print(f"Collecting images for: {label}")
print(f"Existing images: {img_count}")
print()
print("S = Save image")
print("N = Change alphabet")
print("Q = Quit")



# CAMERA

cap = cv2.VideoCapture(0)



# MAIN LOOP

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to grab frame.")

        break


    # Mirror effect
    frame = cv2.flip(frame, 1)



    clean_frame = frame.copy()


    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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


        
        # Bounding box

        padding = 30


        x_min = max(min(x_list) - padding, 0)
        y_min = max(min(y_list) - padding, 0)

        x_max = min(max(x_list) + padding, w)
        y_max = min(max(y_list) + padding, h)

        cropped_hand = clean_frame[
            y_min:y_max,
            x_min:x_max
        ]


        # Draw bounding box for DISPLAY ONLY

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )


        
        # Draw MediaPipe landmarks for DISPLAY ONLY
        

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


    # DISPLAY INFORMATION

    cv2.putText(
        frame,
        f"Letter: {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )


    cv2.putText(
        frame,
        f"Images: {img_count}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "S = Save | N = New Letter | Q = Quit",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show camera window
    cv2.imshow("Sign Language Dataset Collection", frame)


    # KEYBOARD
    

    key = cv2.waitKey(1) & 0xFF


    
    # SAVE

    if key == ord("s"):

        if cropped_hand is not None:

            # Resize clean hand crop
            save_image = cv2.resize(
                cropped_hand,
                (300, 300)
            )


            filename = os.path.join(
                dataset_path,
                f"{label}_{img_count}.jpg"
            )


            cv2.imwrite(
                filename,
                save_image
            )


            print(f"Saved: {filename}")


            img_count += 1


        else:

            print("No hand detected. Image not saved.")


    # CHANGE LETTER

    elif key == ord("n"):

        label = input("\nEnter new alphabet (A-Z): ").upper()


        dataset_path = get_dataset_path(label)


        img_count = len([
            f for f in os.listdir(dataset_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])


        print(f"\nNow collecting: {label}")


    # QUIT

    elif key == ord("q"):

        print("Stopping dataset collection...")

        break


# CLEANUP


cap.release()

cv2.destroyAllWindows()

hands.close()
