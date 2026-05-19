import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open Webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access webcam")
    exit()

# Get Screen Resolution
screen_width, screen_height = pyautogui.size()

# Mouse Smoothing Variables
previous_x = 0
previous_y = 0
smoothening = 5

# Eye Landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Blink Settings
BLINK_THRESHOLD = 0.015
CLICK_DELAY = 1
last_click_time = 0

print("Hands-Free Mouse Control Started")
print("Press Q to exit")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        iris_points = landmarks[474:478]

        for index, point in enumerate(iris_points):
            x = int(point.x * frame_width)
            y = int(point.y * frame_height)

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if index == 1:
                screen_x = screen_width / frame_width * x
                screen_y = screen_height / frame_height * y

                current_x = previous_x + (screen_x - previous_x) / smoothening
                current_y = previous_y + (screen_y - previous_y) / smoothening

                pyautogui.moveTo(current_x, current_y)

                previous_x = current_x
                previous_y = current_y

        left_eye_top = landmarks[LEFT_EYE_TOP]
        left_eye_bottom = landmarks[LEFT_EYE_BOTTOM]

        eye_distance = abs(left_eye_top.y - left_eye_bottom.y)

        current_time = time.time()

        if eye_distance < BLINK_THRESHOLD:
            if current_time - last_click_time > CLICK_DELAY:
                pyautogui.click()
                print("Mouse Clicked")
                last_click_time = current_time

    cv2.imshow("Hands-Free Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
