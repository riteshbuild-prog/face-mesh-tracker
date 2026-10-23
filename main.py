import os
import cv2
import traceback

def run_face_tracker():
    try:
        # Load cascade models from the current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        face_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(base_dir, "haarcascade_eye.xml")

        if not os.path.exists(face_path) or not os.path.exists(eye_path):
            print("[!] XML files not found in the project folder. Run the curl commands first.")
            return

        face_cascade = cv2.CascadeClassifier(face_path)
        eye_cascade = cv2.CascadeClassifier(eye_path)

        if face_cascade.empty() or eye_cascade.empty():
            print("[!] Cascade Classifier failed to initialize from XML files.")
            return

        # Initialize webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[*] Trying standard video backend...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[!] Error: Camera device could not be opened.")
            return

        window_name = "Face Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("[*] Face tracker active. Press 'q' inside the video window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "TARGET ACQUIRED",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8)

                for (ex, ey, ew, eh) in eyes:
                    center = (ex + ew // 2, ey + eh // 2)
                    radius = int(round((ew + eh) * 0.25))
                    cv2.circle(roi_color, center, radius, (255, 200, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception:
        print("[!] Encountered an unhandled exception:")
        traceback.print_exc()

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[*] Tracker closed successfully.")

if __name__ == "__main__":
    run_face_tracker()