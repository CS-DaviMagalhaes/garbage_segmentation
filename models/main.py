from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("../models/keras_v2_100_ep.h5", compile=False)

# Load the labels
class_names = open("../models/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Initialize variables for tracking
confidence_threshold = 0.80  # 80% confidence
time_threshold = 2.0  # 2 seconds
start_time = None
current_class = None

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image, verbose=0) #Put verbose=0 to stop keras printing on console
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    classes = {"0 Plastic":0, "1 Glass":1, "2 Paper":2, "3 Metal":3, "4 Cardboard":3} #metal and cardboard are considered the same class

    with open("gaa.txt", 'w') as file:
        if confidence_score >= confidence_threshold:
            if current_class == class_name:
                # If yes, check the time since the start of high-confidence
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_threshold:
                    print(f"Confirmed Class: {class_name} with {confidence_score * 100:.2f}% confidence")
                    file.write(str(classes[class_name.strip()]))
                    
            else:
                # If it's a new class, reset the timer
                current_class = class_name
                start_time = time.time()
        else:
            # Reset if confidence falls below threshold
            current_class = None
            start_time = None

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()