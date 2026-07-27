import cv2
import os
import mediapipe as mp



# path to save the images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#mediapipe initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)
def get_dataset_path(label):
    path = os.path.join(BASE_DIR, "dataset", label)
    os.makedirs(path, exist_ok=True)
    return path
label =input("Enter the label for the images (A-Z): ").upper()
dataset_path = get_dataset_path(label)

#counter for images
img_count=len(os.listdir(dataset_path))
print(f"\nCollecting images for: {label}")
print("Press S = Save")
print("Press N = Change Alphabet")
print("Press Q = Quit")

#start video capture
cap= cv2.VideoCapture(0)


while True:
    ret , frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break


    #flip image for mirror effect
    
    frame = cv2.flip(frame,1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    cropped_hand = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get bounding box coordinates
            h, w, c = frame.shape
            x_list =[]
            y_list =[]

            for lm in hand_landmarks.landmark:
                x=int(lm.x * w)
                y=int(lm.y * h)

                x_list.append(x)
                y_list.append(y)

            # Add some padding to the bounding box
            padding = 30
            x_min = max(min(x_list) - padding, 0)
            y_min = max(min(y_list) - padding, 0)
            x_max = min( max(x_list) + padding, w)
            y_max = min( max(y_list) + padding, h)

            # Crop the hand region from the frame
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            cropped_hand = frame[y_min:y_max, x_min:x_max]
    
 # Display current label
    cv2.putText(frame, f"Label: {label}", (10, 35),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    #show count 
    cv2.putText(frame,f"Images: {img_count}",(10,75),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
     # Instructions
    cv2.putText(frame, "Press A-Z to change label", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press S to save | Q to quit", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    #show webcam feed
    cv2.imshow("Collecting Images", frame)

    
    #wait for key press
    key = cv2.waitKey(1) & 0xFF
    

    # Save image
    if key == ord('s'):

        if cropped_hand is not None:
            save_img = cv2.resize(cropped_hand, (300, 300))
            filename = os.path.join(
                dataset_path, f"{label}_{img_count}.jpg"
                )

            cv2.imwrite(filename, save_img)

            print(f"Saved {filename}")

            img_count += 1
       
        else:
            print("No hand detected!")
    # Change label
    elif key == ord('n'):

        label = input("\nEnter new alphabet: ").upper()

        dataset_path = get_dataset_path(label)

        img_count = len(os.listdir(dataset_path))
    #quit the program
    elif key == ord('q'):
        break

#realase the webcam and close windows
cap.release()
cv2.destroyAllWindows()  
