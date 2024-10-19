import onnxruntime as ort
import numpy as np
import cv2
from tkinter import *
from PIL import ImageGrab, Image
import PIL.ImageOps

# Load ONNX model
onnx_model_path = 'mnist-8.onnx'
session = ort.InferenceSession(onnx_model_path)

# Preprocess the image before feeding to the model
def preprocess(image):
    # Resize to 28x28 and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    
    # Invert the colors (white background, black digit)
    image = PIL.ImageOps.invert(image)
    
    # Convert to a numpy array
    image = np.array(image).astype(np.float32)
    
    # Normalize the image values to 0-1
    image = image / 255.0
    
    # Reshape the array to match model input shape (1, 1, 28, 28)
    image = image.reshape(1, 1, 28, 28).astype(np.float32)
    
    return image

# Use the ONNX model for prediction
def predict(image):
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    predicted_class = np.argmax(outputs)
    return predicted_class

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)

# Function to capture the canvas content and predict the drawn digit
def recognize_digit():
    # Get the canvas coordinates and capture the image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    
    # Grab the image from the screen
    image = ImageGrab.grab().crop((x, y, x1, y1))
    
    # Preprocess and predict
    processed_image = preprocess(image)
    prediction = predict(processed_image)
    
    # Display the predicted number
    result_label.config(text=f"Predicted Digit: {prediction}")

# Initialize the GUI
root = Tk()
root.title("Draw a Digit")

# Create a canvas to draw on
canvas = Canvas(root, width=200, height=200, bg="white")
canvas.pack()

# Bind mouse events to the canvas
canvas.bind("<B1-Motion>", paint)

# Create a button to recognize the digit
recognize_button = Button(root, text="Recognize Digit", command=recognize_digit)
recognize_button.pack()

# Create a button to clear the canvas
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Label to display the result
result_label = Label(root, text="Predicted Digit: None", font=("Helvetica", 18))
result_label.pack()

# Start the GUI loop
root.mainloop()
