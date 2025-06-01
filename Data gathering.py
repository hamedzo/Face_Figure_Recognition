import mediapipe as mp
import cv2
import os
import pandas as pd
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
# initiate holistic model

class_name = "Happy"

cap = cv2.VideoCapture(0)
 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make Detection
        results = holistic.process(image)
     #   print(results.pose_landmarks)
        
        # recolor image back to bgr for rendering   
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        
        
        
        num_coord = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
        landmark = ['class']
        for val in range(1, num_coord+1):
            landmark += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        # Export coordinte
        try:
            # Extract pose landmark
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
          
            # Extract face landmark
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            
            # concate rows
            row  = pose_row+face_row
            row.insert(0, class_name)

            # Export to CSV
            with open('coords.csv', mode= 'a', newline= '') as f:
                csv_writer = csv.writer(f, delimiter= ',', quotechar= '"', quoting= csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            

        except:
            pass
        cv2.imshow('Holistic Model Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break      
cap.release()
cv2.destroyAllWindows()