
#importing  libraries

import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import vlc
import sys
from twilio.rest import Client

def yawn(mouth):
     return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], 
                            dtype="double")
    
    # 3D model points.
    model_points = np.array(
                           [
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ]
                            )
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    
     # Assuming no lens distortion
    
    dist_coeffs = np.zeros((4,1)) 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, 
                                                                  camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

 #EAR -> Eye Aspect ratio
def ear(eye):
     return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

 # open_avg = train.getAvg()
 # close_avg = train.getAvg()

alert = vlc.MediaPlayer('alert-sound.mp3')
frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5

 #(close_avg+open_avg)/2.0

close_thresh = 0.3
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

 # print(close_thresh)

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while(True):
    ret, frame = capture.read()
    size = frame.shape

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = frame
    rects = detector(gray, 0)

    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))

         # lenLeftEyeX = landmarks[leftEyeIndex[3]][0] - landmarks[leftEyeIndex[0]][0]
        
    # lenLeftEyeY = landmarks[leftEyeIndex[3]][1] - landmarks[leftEyeIndex[0]][1]

    # lenLeftEyeSquared = (lenLeftEyeX ** 2) + (lenLeftEyeY ** 2)
        
    # eyeRegionCount = cv2.countNonZero(mask)

    # normalizedCount = eyeRegionCount/np.float32(lenLeftEyeSquared)

    #############################################################################

        #Get the left eye aspect ratio

        leftEAR = ear(leftEye) 

        #Get the right eye aspect ratio

        rightEAR = ear(rightEye) 

        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        #yawn detection

        if(yawn(shape[mStart:mEnd])>0.6):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            yawn_countdown=1

        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (0,255,255)
            print(flag)

            if(yawn_countdown and flag>=frame_thresh_3):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy after yawn", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
                alert.play()

                if(map_flag):
                    map_flag = 0
                    map_counter+=1

            elif(flag>=frame_thresh_2 and getFaceDirection(shape, size)<0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "*************** Sleep Alert ***************", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
                alert.play()

                if(map_flag):
                    map_flag = 0
                    map_counter+=1

            elif(flag>=frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Drowsy !", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
                alert.play()

                if(map_flag):
                    map_flag = 0
                    map_counter+=1

        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            alert.stop()
            yawn_countdown=0
            map_flag=1
            flag=0

        #sending message using twilio  
        #setting the eyy threshold value   

        if(flag>=162):
            map_counter=0
            SID='ACc8d69b34bc9beb749cb45facc04d0632'
            token='33dc9be8ec04e2da473ea3368a67fc9a'
            ct=Client(SID,token)
            msg='driver doesnt seem ok,please check! or track his gps location'
            ct.messages.create(body=msg,
                               from_='+12019924924',to='+916480502407')
            sys.exit()
       
        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        

    if(avgEAR>close_thresh):
        alert.stop()
    cv2.imshow('Driver', gray)

    if(cv2.waitKey(1)==27):
        break
    
        
capture.release()
cv2.destroyAllWindows()
 

