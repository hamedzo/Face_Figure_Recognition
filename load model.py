import pickle
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'body_language.pkl' not found. Make sure you have trained and saved the model first.")
    exit()

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detection
        results = holistic.process(image)
        
        # recolor image back to bgr for rendering   
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        # 1. Draw Face landmarks    
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1))
        
        # 2. right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(50,200,50), thickness=2, circle_radius=2))
        
        # 3. left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(50,200,50), thickness=2, circle_radius=2))
        
        # 4. pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,100,0), thickness=3, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(255,0,100), thickness=2, circle_radius=2))   

        # Make predictions
        try:
            # Check if required landmarks are detected
            if results.pose_landmarks and results.face_landmarks:
                
                # Extract pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
              
                # Extract face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Extract left hand landmarks (if available)
                if results.left_hand_landmarks:
                    left_hand = results.left_hand_landmarks.landmark
                    left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                else:
                    left_hand_row = [0.0] * (21 * 4)  # Fill with zeros if not detected
                
                # Extract right hand landmarks (if available)
                if results.right_hand_landmarks:
                    right_hand = results.right_hand_landmarks.landmark
                    right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                else:
                    right_hand_row = [0.0] * (21 * 4)  # Fill with zeros if not detected
                
                # Combine all landmarks
                row = pose_row + face_row
                
                # Convert to numpy array for prediction (avoids feature name issues)
                
                X = np.array([row])
                
                # Make prediction
                body_language_class = model.predict(X)[0]
                
                # Get prediction probabilities (if your model supports it)
                try:
                    body_language_prob = model.predict_proba(X)[0]
                    confidence = np.max(body_language_prob)
                    #print(f"Predicted: {body_language_class} (Confidence: {confidence:.2f})")
                    
                    # Display prediction on image
                    cv2.rectangle(image, (0, 0), (300, 80), (245, 117, 16), -1)
                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(confidence, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except AttributeError:
                    # Model doesn't support predict_proba
                    print(f"Predicted: {body_language_class}")
                    
                    # Display prediction on image (without probability)
                    cv2.rectangle(image, (0, 0), (300, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            else:
                # Display message when landmarks are not detected
                cv2.putText(image, 'Detecting landmarks...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            cv2.putText(image, 'Prediction Error', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Body Language Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break      

cap.release()
cv2.destroyAllWindows()