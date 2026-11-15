
import cv2

# Connect to the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the webcam.")
    print("Check your camera connection and permissions.")
    exit()

print("Webcam connected successfully!")
print("Press Q to close the camera window.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not capture a frame.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("SignBridge - Webcam Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed!")
