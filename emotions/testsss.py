import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

# Load the TensorFlow Lite model
model_path = r'C:\Users\AA TRADERS\PycharmProjects\djangoProject\emotions\model (1).tflite1'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the MTCNN detector with adjusted parameters
detector = MTCNN(min_face_size=25, scale_factor=0.9)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to predict emotion for a single face
def predict_emotion(face):
    # Preprocess the face image (resize and normalize)
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype(np.float32) / 255.0  # Convert to FLOAT32 and normalize to [0, 1]
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)  # Add batch dimension

    # Make emotion prediction using TFLite model
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    emotion_prediction = interpreter.get_tensor(output_details[0]['index'])
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]

    # Return the emotion label (e.g., "Happy", "Sad", etc.)
    return emotion_label

# Load an image containing multiple faces
image_path = 'testing.jpg'  # Update with your image path
image = cv2.imread(image_path)

# Detect faces using MTCNN
faces = detector.detect_faces(image)

# List to store predicted emotions for each detected face
predicted_emotions = []

# Iterate through detected faces and predict emotions using TFLite model
for result in faces:
    x, y, width, height = result['box']
    face = image[y:y+height, x:x+width]
    emotion_label = predict_emotion(face)

    # Append the predicted emotion to the list
    predicted_emotions.append(emotion_label)

    # Draw a rectangle around the face with the emotion label
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 1)
    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

# Display the predicted emotions for each detected face
print("Predicted Emotions for each face:")
for i, emotion in enumerate(predicted_emotions):
    print( emotion)

# Save the image with emotion predictions
cv2.imwrite('path/to/your/output/image_with_emotions.jpg', image)

# Display the saved image with emotion predictions using OpenCV imshow
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
