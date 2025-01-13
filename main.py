import cv2
import dlib
import numpy as np
import os

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

# Load frames (PNG with transparency)
frames_path = "./frame"
frames = [
    cv2.imread(os.path.join(frames_path, f), cv2.IMREAD_UNCHANGED)
    for f in os.listdir(frames_path)
    if f.endswith('.png')
]

# Ensure frames are loaded
if not frames:
    print("Error: No valid PNG frames found in the 'frame/' directory. Please add PNG images and try again.")
    exit()

# Resize frame to fit the face, including stems
def resize_frame(frame, landmarks, scale_factor=1.0):
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Calculate the face width using landmarks (jawline: points 0 and 16)
    face_width = np.linalg.norm(landmarks[0] - landmarks[16])

    # Calculate scaling factor to fit the frame width to the face width
    scale_factor *= face_width / frame_width
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Position the frame based on landmarks
    left_eye = np.mean(landmarks[36:42], axis=0)  # Average left eye position
    right_eye = np.mean(landmarks[42:48], axis=0)  # Average right eye position
    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)

    y_offset = int(new_height * 0.5)  # Adjust offset for glasses above the eyes
    x = eye_center_x - new_width // 2
    y = eye_center_y - y_offset

    return resized_frame, (x, y)

# Overlay frame on the image
def overlay_frame(image, frame, position):
    x, y = position
    h, w = frame.shape[:2]

    # Ensure the frame fits within the image bounds
    if y < 0 or x < 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return  # Skip overlay if out of bounds

    # Blend the frame with the image
    if frame.shape[2] == 4:  # RGBA frame
        alpha = frame[:, :, 3] / 255.0  # Alpha channel as mask
        for c in range(0, 3):  # Blend BGR channels
            image[y:y+h, x:x+w, c] = (
                alpha * frame[:, :, c] + (1 - alpha) * image[y:y+h, x:x+w, c]
            )
    else:
        image[y:y+h, x:x+w] = frame  # No transparency, direct overlay

# Webcam Mode
def webcam_mode():
    cap = cv2.VideoCapture(0)
    current_frame_idx = 0
    scale_factor = 1.0  # Initial scale factor
    stem_x_adjust = 0  # Adjustment for x position (left/right)
    stem_y_adjust = 0  # Adjustment for y position (up/down)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Resize and position the frame
            glasses_frame, position = resize_frame(
                frames[current_frame_idx], landmarks, scale_factor
            )

            # Adjust frame position for stem alignment
            adjusted_position = (
                position[0] + stem_x_adjust,  # Adjust x (left/right)
                position[1] + stem_y_adjust   # Adjust y (up/down)
            )
            overlay_frame(frame, glasses_frame, adjusted_position)

        # Display instructions
        cv2.putText(frame, "Press + to increase size, - to decrease size", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Arrow Keys: Adjust frame position", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press N to change frame, ESC to exit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Virtual Frame Try-On", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('n'):  # 'n' to switch to the next frame
            current_frame_idx = (current_frame_idx + 1) % len(frames)
        elif key == ord('+'):  # '+' to increase frame size
            scale_factor += 0.1
        elif key == ord('-'):  # '-' to decrease frame size
            scale_factor = max(0.1, scale_factor - 0.1)  # Ensure scale doesn't go below 0.1
        elif key == 82:  # Arrow Up
            stem_y_adjust -= 5
        elif key == 84:  # Arrow Down
            stem_y_adjust += 5
        elif key == 81:  # Arrow Left
            stem_x_adjust -= 5
        elif key == 83:  # Arrow Right
            stem_x_adjust += 5

    cap.release()
    cv2.destroyAllWindows()

# Main Menu
def main():
    print("Select an option:")
    print("1. Webcam Mode")
    choice = input("Enter your choice: ")

    if choice == "1":
        webcam_mode()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
