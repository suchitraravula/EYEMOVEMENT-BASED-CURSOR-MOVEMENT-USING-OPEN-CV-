import cv2
import mediapipe as mp
import pyautogui

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # *Right Eye Landmark (User's Left Eye)*
        right_eye = landmarks[474]  
        right_x = int(right_eye.x * frame_w)
        right_y = int(right_eye.y * frame_h)
        cv2.circle(frame, (right_x, right_y), 3, (255, 0, 0), -1)  # Blue dot

        # *Left Eye Landmark (User's Right Eye)*
        left_eye = landmarks[469]  
        left_x = int(left_eye.x * frame_w)
        left_y = int(left_eye.y * frame_h)
        cv2.circle(frame, (left_x, left_y), 3, (0, 255, 0), -1)  # Green dot

        # *Cursor Movement - Use Average of Both Eyes*
        screen_x = screen_w * ((right_eye.x + left_eye.x) / 2)
        screen_y = screen_h * ((right_eye.y + left_eye.y) / 2)
        pyautogui.moveTo(screen_x, screen_y)

        # *Blink Detection for Click*
        left_blink = [landmarks[145], landmarks[159]]  # Left eye blink detection
        right_blink = [landmarks[374], landmarks[386]]  # Right eye blink detection

        for landmark in left_blink + right_blink:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots for blink detection

        # If eye is closed (blink detected), perform a click
        if (left_blink[0].y - left_blink[1].y) < 0.004 or (right_blink[0].y - right_blink[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow("Eye Controlled Mouse", frame)

    # Press "Esc" key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()