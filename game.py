import cv2
import mediapipe as mp
import numpy as np

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Canvas to draw on
current_drawing = []  # Store the drawn path (fingertip positions)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand tracking
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distances to check if hand is open or closed
            thumb_index_distance = calculate_distance(index_finger_tip, thumb_tip)
            index_middle_distance = calculate_distance(index_finger_tip, middle_finger_tip)

            # Threshold for detecting if the hand is open (you may need to tune this)
            open_hand_threshold = 0.1

            # Only track the index finger if the hand is open
            if thumb_index_distance > open_hand_threshold and index_middle_distance > open_hand_threshold:
                x = int(index_finger_tip.x * 640)
                y = int(index_finger_tip.y * 480)

                # Add the fingertip position to the drawing path
                current_drawing.append((x, y))

                # Draw the path on the canvas (only track index finger)
                if len(current_drawing) > 1:
                    for i in range(len(current_drawing) - 1):
                        cv2.line(drawing_canvas, current_drawing[i], current_drawing[i + 1], (255, 255, 255), 3)

            # Do not draw hand landmarks, only the index finger path is tracked and displayed

    # Combine live video feed with drawing canvas
    combined_frame = cv2.addWeighted(frame, 0.7, drawing_canvas, 0.3, 0)

    cv2.imshow('Air Calculator - Index Finger Tracking', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
