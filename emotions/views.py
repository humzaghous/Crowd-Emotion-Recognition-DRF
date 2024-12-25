from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf

class EmotionRecognitionAPIView(APIView):
    def post(self, request):
        try:
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

            # Process the image
            if 'image' not in request.FILES:
                return Response({'error': 'Image file not found in request.'}, status=status.HTTP_400_BAD_REQUEST)

            # Decode the image
            image = cv2.imdecode(np.frombuffer(request.FILES['image'].read(), np.uint8), cv2.IMREAD_COLOR)

            # List to store predicted emotions
            predicted_emotions = []

            # Iterate through detected faces and predict emotions using TFLite model
            faces = detector.detect_faces(image)
            for result in faces:
                x, y, width, height = result['box']
                face = image[y:y+height, x:x+width]
                emotion_label = self.predict_emotion_tflite(interpreter, input_details, output_details, face, emotion_labels)

                # Append predicted emotion to the list
                predicted_emotions.append(emotion_label)

            # Return the response with the predicted emotions list
            response_data = {'emotions': predicted_emotions}
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            # Log the specific exception for debugging
            print(f"Exception: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def predict_emotion_tflite(self, interpreter, input_details, output_details, face, emotion_labels):
        # Preprocess the face image (resize and normalize)
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face / 255.0  # Normalize to [0, 1]
        face = np.float32(face)  # Convert to float32
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], face)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        emotion_prediction = interpreter.get_tensor(output_details[0]['index'])
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        return emotion_label
