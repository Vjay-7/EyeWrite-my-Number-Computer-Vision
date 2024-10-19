import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import warnings

# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load ONNX model
onnx_model_path = 'mnist-8.onnx'  # Ensure the correct path to the ONNX model
session = ort.InferenceSession(onnx_model_path)

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Canvas to draw on
drawing_canvas[:, :] = (255, 255, 255)  # Set the entire canvas to white

current_drawing = []  # Store the drawn path (fingertip positions)
recognized_digit_text = "Recognized digit: "  # To store the recognized digit for display

# Set the drawing box coordinates (fixed size)
box_top_left = (200, 100)
box_bottom_right = (440, 340)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# Function to preprocess and recognize the digit using the ONNX model
def recognize_digit(canvas):
    # Extract the drawing area from the canvas
    drawing_area = canvas[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]

    # Convert to grayscale and threshold the image to make the drawn lines clearer
    gray_image = cv2.cvtColor(drawing_area, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours to help center the drawing
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])

        # Extract only the drawn digit
        digit = thresh_image[y:y+h, x:x+w]

        # Resize the extracted digit to fit into 20x20, leaving some padding around
        digit = cv2.resize(digit, (20, 20))

        # Create a 28x28 image and place the resized digit in the center
        centered_digit = np.zeros((28, 28), dtype=np.uint8)
        centered_digit[4:24, 4:24] = digit  # Center the 20x20 digit

        # Show the final preprocessed image (for debugging)
        cv2.imshow('Preprocessed Image for MNIST', centered_digit)

        # Normalize and reshape the image for the model
        normalized_image = centered_digit.astype(np.float32) / 255.0
        normalized_image = normalized_image.reshape(1, 1, 28, 28)  # MNIST format: (1, 1, 28, 28)

        # Run inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: normalized_image})

        # Check if the output result is valid and return the predicted digit
        if result:
            predicted_digit = np.argmax(result[0])
            return predicted_digit
        else:
            print("No output from model.")
            return None
    else:
        print("No contour found.")
        return None

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

            # Threshold for detecting if the hand is open (tune as necessary)
            open_hand_threshold = 0.1

            # Only track the index finger if the hand is open
            if thumb_index_distance > open_hand_threshold and index_middle_distance > open_hand_threshold:
                x = int(index_finger_tip.x * 640)
                y = int(index_finger_tip.y * 480)

                # Only track the finger if it's inside the drawing box
                if box_top_left[0] < x < box_bottom_right[0] and box_top_left[1] < y < box_bottom_right[1]:
                    current_drawing.append((x, y))

                    # Draw the path in red on a separate canvas
                    if len(current_drawing) > 1:
                        for i in range(len(current_drawing) - 1):
                            cv2.line(drawing_canvas, current_drawing[i], current_drawing[i + 1], (0, 0, 255), 3)

    # Draw the bounding box where the user should draw (green outline)
    cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 255, 0), 2)

    # Create a 70% opaque white background inside the drawing box
    white_box = np.full((box_bottom_right[1] - box_top_left[1], box_bottom_right[0] - box_top_left[0], 3), (255, 255, 255), dtype=np.uint8)
    frame[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]] = cv2.addWeighted(
        frame[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]], 0.3,
        white_box, 0.7, 0)

    # Overlay the drawing path onto the frame
    for i in range(len(current_drawing) - 1):
        cv2.line(frame, current_drawing[i], current_drawing[i + 1], (0, 0, 255), 3)

    # Display the recognized digit at the bottom of the screen
    cv2.putText(frame, recognized_digit_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  
    cv2.imshow('Air Calculator - Index Finger Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if len(current_drawing) > 1:
            predicted_digit = recognize_digit(drawing_canvas)
            if predicted_digit is not None:
                recognized_digit_text = f"Recognized digit: {predicted_digit}"
            else:
                recognized_digit_text = "Recognized digit: Failed to recognize."

        # Reset the drawing
        current_drawing = []
        drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        drawing_canvas[:, :] = (255, 255, 255)  # Reset the drawing canvas to white

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
