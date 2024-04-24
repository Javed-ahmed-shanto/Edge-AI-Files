

import cv2

def test_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frames.")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if name == "main":
    test_camera(0)  # Change the camera index if needed
