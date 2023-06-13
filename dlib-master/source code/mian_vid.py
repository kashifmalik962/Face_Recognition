import cv2
import numpy as np

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect and draw faces
def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw the faces and overlay the icon
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the center coordinates of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the radius of the circle
        radius = min(w, h) // 2

        # Draw a circle as the icon
        cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), 2)

    return image

# Load an image
image = cv2.imread('Messi1.webp')

# Detect faces and overlay the icon
result = detect_faces(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
