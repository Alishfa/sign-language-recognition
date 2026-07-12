import cv2
import os



current_label = "A"


# path to save the images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Base directory: {BASE_DIR}")
def get_dataset_path(label):
    path = os.path.join(BASE_DIR, "dataset", label)
    os.makedirs(path, exist_ok=True)
    return path
dataset_path = get_dataset_path(current_label)

#counter for images
img_count=len(os.listdir(dataset_path))

#start video capture
cap= cv2.VideoCapture(0)


while True:
    ret , frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break


    #flip image for mirror effect
    
    frame = cv2.flip(frame,1)
    
    #draw rectangle for hand region
    
    cv2.rectangle(frame,(300,100),(600,400),(0,255,0),2)
    
 # Display current label
    cv2.putText(frame, f"Label: {current_label}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    #show count 
    cv2.putText(frame,f"Images: {img_count}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
     # Instructions
    cv2.putText(frame, "Press A-Z to change label", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press S to save | Q to quit", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    #show webcam feed
    cv2.imshow("Collecting Images", frame)

    
    #wait for key press
    key = cv2.waitKey(1) 
    
    #quit
    if key == ord('q'):
        break

    # Save image
    elif key == ord('s'):

        
        # Crop ROI
        roi = frame[100:400, 300:600]

        # Create image name
        img_name = os.path.join(
            dataset_path,
            f"{current_label}_{img_count}.jpg"
        )


        cv2.imwrite(img_name, roi)

        print(f"Saved {img_name}")

        img_count += 1
       
        # Change label dynamically
    elif ord('A') <= key <= ord('Z'):

        current_label = chr(key)

        dataset_path = get_dataset_path(current_label)

        img_count = len(os.listdir(dataset_path))

        print(f"Switched to label: {current_label}")


cap.release()
cv2.destroyAllWindows()  
