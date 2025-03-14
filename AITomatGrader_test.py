import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Adjust based on real-world reference
def estimate_diameter(image_path, pixel_to_cm_ratio=0.03):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing: Blur + Adaptive Threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reduce minimum area filtering to detect smaller objects
    min_contour_area = 200  # Adjust based on testing
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Instead of bounding box, use min enclosing circle for better accuracy
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter_pixels = 2 * radius  # Diameter = 2 * radius
        diameter_cm = diameter_pixels * pixel_to_cm_ratio
        return diameter_cm
    else:
        return None


def get_color_percentage(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(img, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(img, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2
    green_mask = cv2.inRange(img, green_lower, green_upper)
    yellow_mask = cv2.inRange(img, yellow_lower, yellow_upper)

    # Calculate raw pixel counts
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    yellow_pixels = np.sum(yellow_mask > 0)

    # Total detected color pixels
    total_color_pixels = red_pixels + green_pixels + yellow_pixels

    # Avoid division by zero
    if total_color_pixels == 0:
        return 0, 0, 0  # No colors detected

    # Normalize percentages to sum up to 100%
    red_percentage = (red_pixels / total_color_pixels) * 100
    green_percentage = (green_pixels / total_color_pixels) * 100
    yellow_percentage = (yellow_pixels / total_color_pixels) * 100

    return red_percentage, green_percentage, yellow_percentage


def predict_tomato(img_path):
    # Load and preprocess image for CNN
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load the saved model
    model_path = os.path.join('models', 'AI_TomatGrader.keras')
    model = load_model(model_path, compile=False)  # Load without optimizer
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Predict class using CNN
    prediction = model.predict(img_array)
    class_names = ["Reject", "Ripe", "Unripe"]
    predicted_class = class_names[np.argmax(prediction)]

    # Extract diameter & color using your finalized code
    diameter = estimate_diameter(img_path)  # Your function
    red, green, yellow = get_color_percentage(img_path)  # Your function

    # Print results
    print(f"Tomato Condition: {predicted_class}")
    print(
        f"Color Percentage: {red:.2f}% Red, {green:.2f}% Green, {yellow:.2f}% Yellow")
    print(f"Estimated Diameter: {diameter:.2f} cm\n")


predict_tomato("healthy (79).jpg")
