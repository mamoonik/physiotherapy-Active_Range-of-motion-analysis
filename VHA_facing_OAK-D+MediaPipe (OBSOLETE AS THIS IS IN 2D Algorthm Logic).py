import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

import depthai as dai
from skimage.metrics import structural_similarity as ssim
import keyboard #load keyboard package
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
## Initializing mp_pose for Pose capture
#######################################################################################
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, model_complexity=2, smooth_landmarks =True)
holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.4)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence = 0.7)

##################################################################
##################################################################
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(image)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
   # Initialize a list to store the detected landmarks.
    landmarks = []
    landmarks_world = []
    # print(height, width)
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        # mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_world_landmarks,
        #                           connections=mp_pose.POSE_CONNECTIONS)
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            
        for landmark in results.pose_world_landmarks.landmark:
            # # Append the landmark into the list.
            landmarks_world.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # # Display the original input image and the resultant image.
        # plt.figure(figsize=[22,22])
        # plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        # plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return output_image, landmarks, landmarks_world

    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks, landmarks_world
    
def calculateAngle(landmark1, landmark2, landmark3):
    
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle
def calculateAngle_relative3D(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''
    # Get the required landmarks coordinates.
    x1, y1, z1 = landmark1
    x2, y2, z2 = landmark2
    x3, y3, z3 = landmark3
    # print(x1,y1,z1)
    # print(x2,y2,z2)
    # print(x3,y3,z3)
    vec_1 = (x1 - x2, y1 - y2, z1 - z2)
    vec_2 = (x3 - x2, y3 - y2, z3 - z2)
    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    dot_product_vec1and2 = np.dot(vec_1,vec_2)
    # mag_vec_1 = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )
    mag_vec_1 = np.linalg.norm(vec_1)
    mag_vec_2 = np.linalg.norm(vec_2)
    # mag_vec_2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
    angle = math.degrees( math.acos (dot_product_vec1and2 / (mag_vec_1 * mag_vec_2) ) )
     
    # Calculate the angle between the three points
    # angle = math.degrees(math.acos2(y3 - y2, x3 - x2) - math.acos2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    # if angle < 0:
    #     # Add 360 to the found angle.
    #     angle += 360
    
    # Return the calculated angle.
    return angle
##################################################################
##################################################################
# "l"
def classifyPose_lumbarAROM(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    # left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # # Get the angle between the right shoulder, elbow and wrist points. 
    # right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # # Get the angle between the left elbow, shoulder and hip points. 
    # left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # # Get the angle between the right hip, shoulder and elbow points. 
    # right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

       ##
       ##   
       ##                        
    # Get the angle between the right hip, knee and ankle points 
    right_bending_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
       # Get the angle between the right hip, knee and ankle points 
    left_bending_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])



    ##
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] 
    x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_HIP_x,mid_HIP_y,mid_HIP_z = (x3+x4)/2 , (y3+y4)/2, (z3+z4)/2
    
    GROUND_HIP_NOSE_angle = calculateAngle((mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
                                      (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                      landmarks[mp_pose.PoseLandmark.NOSE.value])
    
    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
    lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))


    # Check if it is FORWARD BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_bending_angle < 165 or right_bending_angle < 165 and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 and dist_between_shoulders < lenght_of_body/6:
        # Specify the label of the pose that is tree pose.
        label = 'LUMBAR AROM FORWARD BENDING MOTION - Bending angle:'+ str(left_bending_angle)
        # print (left_bending_angle)
        # print (right_bending_angle)
         
    # Check if it is BACKWARD BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_bending_angle > 190 or right_bending_angle > 190 and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 and dist_between_shoulders < lenght_of_body/6:
        # Specify the label of the pose that is tree pose.
        label = 'LUMBAR AROM BACKWARD BENDING MOTION - Bending angle:' + str(360 -left_bending_angle)
        # print (left_bending_angle)
        # print (right_bending_angle)

    # print (left_bending_angle)

    # Check if it is RIGHT SIDE BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if GROUND_HIP_NOSE_angle < 165  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 and dist_between_shoulders > lenght_of_body/5:
        # Specify the label of the pose that is tree pose.
        label = 'LUMBAR AROM RIGHT BENDING MOTION - Bending angle:' + str(GROUND_HIP_NOSE_angle)
        # print (left_bending_angle)
        # print (right_bending_angle)

        # Check if one leg is straight
    if GROUND_HIP_NOSE_angle > 195  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 and dist_between_shoulders > lenght_of_body/5:
        # Specify the label of the pose that is tree pose.
        label = 'LUMBAR AROM LEFT BENDING MOTION. - Bending angle:' + str(GROUND_HIP_NOSE_angle)
        # print (left_bending_angle)
        # print (right_bending_angle)
    # print (left_bending_angle)
    # if LEFT_MID_SHOULDER_NOSE_angle > 100:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM RIGHT BENDING. - Bending angle:' + str(180 - LEFT_MID_SHOULDER_NOSE_angle)
    # if LEFT_MID_SHOULDER_NOSE_angle < 80:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM LEFT BENDING. - Bending angle:' + str(LEFT_MID_SHOULDER_NOSE_angle)
    
    # if CERVICAL_EXTENSION_FLEXION_angle < 80:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM FLEXION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    # if CERVICAL_EXTENSION_FLEXION_angle > 100:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM EXTENSION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    
    # if CERVICAL_ABDUCTION_LEFT_angle > 150 and CERVICAL_ABDUCTION_RIGHT_angle > 150:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_LEFT_angle) + str(CERVICAL_ABDUCTION_RIGHT_angle)
    #     # label_2 = 'CERVICAL ABDUCTION_RIGHT -  RIGHT SIDE ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_RIGHT_angle)

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
def classifyPose_lumbarAROM_3D(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    # # Calculate the required angles for the ROMS we are interested in.
    # #----------------------------------------------------------------------------------------------------------------
    #     # Get the angle between the left shoulder, elbow and wrist points. 
    # left_elbow_angle = calculateAngle_relative3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 

    # right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # print(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    # print(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    # print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # label = str(360-right_elbow_angle)

    # # Get the angle between the left hip, knee and ankle points. 
    # left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # # Get the angle between the right hip, knee and ankle points 
    # right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])  
  
        
    
    # # Get the angle between the right hip, knee and ankle points 
    # right_bending_angle = calculateAngle_relative3D(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    #    # Get the angle between the right hip, knee and ankle points 
    # left_bending_angle = calculateAngle_relative3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    # print(right_e/lbow_angle)
    # print(left_elbow_angle)

    if mp_pose.PoseLandmark.LEFT_SHOULDER and mp_pose.PoseLandmark.RIGHT_SHOULDER and mp_pose.PoseLandmark.LEFT_HIP and mp_pose.PoseLandmark.RIGHT_HIP and mp_pose.PoseLandmark.LEFT_ANKLE and mp_pose.PoseLandmark.RIGHT_ANKLE:
        # # Get the angle between the left hip, knee and ankle points. 
        # left_knee_angle = calculateAngle_relative3D(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        #                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        #                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # # Get the angle between the right hip, knee and ankle points 
        # right_knee_angle = calculateAngle_relative3D(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        #                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        #                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])  
        # # print(right_knee_angle)
        # print(left_knee_angle)   
        x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
        x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2
        x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] 
        x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        mid_HIP_x,mid_HIP_y,mid_HIP_z = (x3+x4)/2 , (y3+y4)/2, (z3+z4)/2
        x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
        x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z = (x5+x6)/2 , (y5+y6)/2, (z5+z6)/2

        GROUND_HIP_SHOULDER_angle = calculateAngle_relative3D(
                                        (mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
                                        (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                        (mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z))
        
        dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))
        x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
        lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))
        # Check if one leg is straight
        # if left_bending_angle < 165 or right_bending_angle < 165 and dist_between_shoulders < lenght_of_body/6: # and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195
        # if left_bending_angle < 165 or right_bending_angle < 165 and dist_between_shoulders < lenght_of_body/6: # and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 
        # print (GROUND_HIP_SHOULDER_angle)

        if GROUND_HIP_SHOULDER_angle < 160 and dist_between_shoulders < lenght_of_body/6: # and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
                # Specify the label of the pose that is tree pose.
            label = 'LUMBAR AROM FORWARD BENDING MOTION - Bending angle:'+ str(GROUND_HIP_SHOULDER_angle)
            print(label)
            # print (right_bending_angle)
            #   
        # else:
        #     print(GROUND_HIP_SHOULDER_angle)  
        
    else:
        print("BODY NOT IN FRAME!!!")

    # Check if it is BACKWARD BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    # Check if one leg is straight
    # if left_bending_angle > 190 or right_bending_angle > 190 and dist_between_shoulders < lenght_of_body/6: # and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
    
        # Specify the label of the pose that is tree pose.
        # label = 'LUMBAR AROM BACKWARD BENDING MOTION - Bending angle:' + str(360 -left_bending_angle)
        # print (left_bending_angle)
        # print (right_bending_angle)

    # print (left_bending_angle)

    # Check if it is RIGHT SIDE BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    # if GROUND_HIP_NOSE_angle < 165 and dist_between_shoulders > lenght_of_body/5: #  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195
    #     # Specify the label of the pose that is tree pose.
    #     label = 'LUMBAR AROM RIGHT BENDING MOTION - Bending angle:' + str(GROUND_HIP_NOSE_angle)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)

    #     # Check if one leg is straight
    # if GROUND_HIP_NOSE_angle > 195 and dist_between_shoulders > lenght_of_body/5: #  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195 and dist_between_shoulders > lenght_of_body/5:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'LUMBAR AROM LEFT BENDING MOTION. - Bending angle:' + str(GROUND_HIP_NOSE_angle)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)
    

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "c"
def classifyPose_cervicalAROM(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    
    # GROUND_HIP_NOSE_angle = calculateAngle((mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
    #                                   (mid_HIP_x,mid_HIP_y,mid_HIP_z),
    #                                   landmarks[mp_pose.PoseLandmark.NOSE.value])
    # ## CERVICAL AROM LEFT/RIGHT BENDING
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    LEFT_MID_SHOULDER_NOSE_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      (mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z),
                                      landmarks[mp_pose.PoseLandmark.NOSE.value])
    
    ## CERVICAL AROM FLEXION/EXTENSION 
    # x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    # x2,y2,z2 = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    # mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    CERVICAL_EXTENSION_FLEXION_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_EAR.value],
                                      landmarks[mp_pose.PoseLandmark.NOSE.value])

    if LEFT_MID_SHOULDER_NOSE_angle > 100:
        # Specify the label of the pose that is tree pose.
        label = 'CERVICAL AROM RIGHT BENDING. - Bending angle:' + str(180 - LEFT_MID_SHOULDER_NOSE_angle)
    if LEFT_MID_SHOULDER_NOSE_angle < 80:
        # Specify the label of the pose that is tree pose.
        label = 'CERVICAL AROM LEFT BENDING. - Bending angle:' + str(LEFT_MID_SHOULDER_NOSE_angle)
    
    if CERVICAL_EXTENSION_FLEXION_angle < 80:
         # Specify the label of the pose that is tree pose.
        label = 'CERVICAL AROM FLEXION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    if CERVICAL_EXTENSION_FLEXION_angle > 100:
         # Specify the label of the pose that is tree pose.
        label = 'CERVICAL AROM EXTENSION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    
    # if SHOULDER_ABDUCTION_LEFT_angle > 150 and SHOULDER_ABDUCTION_RIGHT_angle > 150:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'SHOULDER AROM ABDUCTION - Bending angle:' + str(SHOULDER_ABDUCTION_LEFT_angle) + str(SHOULDER_ABDUCTION_RIGHT_angle)
    #     # label_2 = 'CERVICAL ABDUCTION_RIGHT -  RIGHT SIDE ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_RIGHT_angle)

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "s"
def classifyPose_shoulderAROM(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    # # Get the angle between the left elbow, shoulder and hip points. 
    # left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # # Get the angle between the right hip, shoulder and elbow points. 
    # right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # # Get the angle between the left hip, knee and ankle points. 
    # left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # # Get the angle between the right hip, knee and ankle points 
    # right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    #    ##
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
    lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))


    SHOULDER_ABDUCTION_LEFT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    SHOULDER_ABDUCTION_RIGHT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    ####
    ####
    ####
    #### ER NEUTRAL POSITION:
    ####
    ####
    ####
    SHOULDER_ER_NEUTRALPOSITION_LEFT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    SHOULDER_ER_NEUTRALPOSITION_RIGHT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # SHOULDER_EXTENSION_ABDUCTION_RIGHT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    # SHOULDER_EXTENSION_ABDUCTION_LEFT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                    
    

    ##
    ##
    ##
    ## Normal Vector calculation done in each classification
    ##
    # ux, uy, uz = u = [x1-mid_HIP_x, y1-mid_HIP_y, z1-mid_HIP_z] #first vector
    # vx, vy, vz = v = [x2-mid_HIP_x, y2-mid_HIP_y, z2-mid_HIP_z] #sec vector
    # u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx] #cross product
    # point  = np.array([mid_HIP_x,mid_HIP_y,mid_HIP_z] )
    # normal = np.array(u_cross_v)
    # d = -point.dot(normal)
    # print('plane equation:\n{:1.4f}x + {:1.4f}y + {:1.4f}z + {:1.4f} = 0'.format(normal[0], normal[1], normal[2], d))
    # xx, yy = np.meshgrid(range(10), range(10))
    # z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    # # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    # plt3d.quiver(mid_HIP_x, mid_HIP_y, mid_HIP_z, normal[0], normal[1], normal[2], color="m")
    # plt3d.plot_surface(xx, yy, z)
    # plt3d.set_xlabel("X", color='red', size=18)
    # plt3d.set_ylabel("Y", color='green', size=18)
    # plt3d.set_zlabel("Z", color='b', size=18)
    # plt.show()
    # print(GROUND_HIP_NOSE_angle)
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------

    # # Check if the both arms are straight.
    # if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

    #     # Check if shoulders are at the required angle.
    #     if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # # Check if it is the warrior II pose.
    # #----------------------------------------------------------------------------------------------------------------

    #         # Check if one leg is straight.
    #         if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

    #             # Check if the other leg is bended at the required angle.
    #             if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

    #                 # Specify the label of the pose that is Warrior II pose.
    #                 label = 'Warrior II Pose' 
                        
    # #----------------------------------------------------------------------------------------------------------------
    
    # # Check if it is the T pose.
    # #----------------------------------------------------------------------------------------------------------------
    
    #         # Check if both legs are straight
    #         if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

    #             # Specify the label of the pose that is tree pose.
    #             label = 'T Pose'

    # #----------------------------------------------------------------------------------------------------------------
    
    # # Check if it is the tree pose.
    # #----------------------------------------------------------------------------------------------------------------
    
    # # Check if one leg is straight
    # if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

    #     # Check if the other leg is bended at the required angle.
    #     if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

    #         # Specify the label of the pose that is tree pose.
    #         label = 'Tree Pose'
                
    # #----------------------------------------------------------------------------------------------------------------
     #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is FORWARD BENDING MOTION.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    # if left_bending_angle < 165 or right_bending_angle < 165 and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'FORWARD BENDING MOTION - Bending angle:'+ str(left_bending_angle)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)
         
    # # Check if it is BACKWARD BENDING MOTION.
    # #----------------------------------------------------------------------------------------------------------------
    
    # # Check if one leg is straight
    # if left_bending_angle > 190 or right_bending_angle > 190 and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'BACKWARD BENDING MOTION - Bending angle:' + str(left_bending_angle - 90)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)

    # # print (left_bending_angle)

    # # Check if it is RIGHT SIDE BENDING MOTION.
    # #----------------------------------------------------------------------------------------------------------------
    
    # # Check if one leg is straight
    # if GROUND_HIP_NOSE_angle < 165  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'RIGHT BENDING MOTION - Bending angle:' + str(GROUND_HIP_NOSE_angle)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)

    #     # Check if one leg is straight
    # if GROUND_HIP_NOSE_angle > 195  and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'LEFT BENDING MOTION. - Bending angle:' + str(GROUND_HIP_NOSE_angle)
    #     # print (left_bending_angle)
    #     # print (right_bending_angle)
    # # print (left_bending_angle)
    # if LEFT_MID_SHOULDER_NOSE_angle > 100:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM RIGHT BENDING. - Bending angle:' + str(180 - LEFT_MID_SHOULDER_NOSE_angle)
    # if LEFT_MID_SHOULDER_NOSE_angle < 80:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM LEFT BENDING. - Bending angle:' + str(LEFT_MID_SHOULDER_NOSE_angle)
    
    # if CERVICAL_EXTENSION_FLEXION_angle < 80:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM FLEXION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    # if CERVICAL_EXTENSION_FLEXION_angle > 100:
    #      # Specify the label of the pose that is tree pose.
    #     label = 'CERVICAL AROM EXTENSION - Bending angle:' + str(360 -CERVICAL_EXTENSION_FLEXION_angle)
    
    if SHOULDER_ABDUCTION_LEFT_angle > 150 and SHOULDER_ABDUCTION_LEFT_angle < 220 and SHOULDER_ABDUCTION_RIGHT_angle > 180 and SHOULDER_ABDUCTION_RIGHT_angle < 270 and dist_between_shoulders > lenght_of_body/5:
         # Specify the label of the pose that is tree pose.
        label = 'SHOULDER AROM ABDUCTION - Bending angle:' + str(360- SHOULDER_ABDUCTION_LEFT_angle) + str(360 - SHOULDER_ABDUCTION_RIGHT_angle)
        # label_2 = 'CERVICAL ABDUCTION_RIGHT -  RIGHT SIDE ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_RIGHT_angle)
    
    if SHOULDER_ABDUCTION_LEFT_angle > 70 and SHOULDER_ABDUCTION_LEFT_angle < 270 and SHOULDER_ABDUCTION_RIGHT_angle > 70 and SHOULDER_ABDUCTION_RIGHT_angle <270 and dist_between_shoulders < lenght_of_body/6:
         # Specify the label of the pose that is tree pose.
        label = 'SHOULDER FLEXION  - Bending angle:' + str(360 - SHOULDER_ABDUCTION_LEFT_angle) + str(360 - SHOULDER_ABDUCTION_RIGHT_angle)
        # print(SHOULDER_ABDUCTION_LEFT_angle)
        # print(SHOULDER_ABDUCTION_RIGHT_angle)
        # print(360-SHOULDER_ABDUCTION_LEFT_angle)
        # print(360-SHOULDER_ABDUCTION_RIGHT_angle)
        # label_2 = 'CERVICAL ABDUCTION_RIGHT -  RIGHT SIDE ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_RIGHT_angle)

    if SHOULDER_ABDUCTION_LEFT_angle > 20 and SHOULDER_ABDUCTION_LEFT_angle <70 and SHOULDER_ABDUCTION_RIGHT_angle > 20 and SHOULDER_ABDUCTION_RIGHT_angle < 70 and dist_between_shoulders < lenght_of_body/6:
         # Specify the label of the pose that is tree pose.
        label = 'SHOULDER EXTENSION  - Bending angle:' + str(SHOULDER_ABDUCTION_LEFT_angle) + str(SHOULDER_ABDUCTION_RIGHT_angle)
        # label_2 = 'CERVICAL ABDUCTION_RIGHT -  RIGHT SIDE ABDUCTION Bending angle:' + str(CERVICAL_ABDUCTION_RIGHT_angle)

    if left_elbow_angle > 60 and left_elbow_angle < 100 and x1 < x2: #and right_elbow_angle > 60 and right_elbow_angle < 100:
        label = 'LEFT_IR IN NEUTRAL  - Bending angle:' + str(left_elbow_angle)

    if right_elbow_angle > 270 and right_elbow_angle < 310 and x1 < x2:
        label = 'RIGHT _IR IN NEUTRAL  - Bending angle:' + str(360-right_elbow_angle)

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "t"
def classifyPose_hipAROM_topView(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

   
    ##############################################################################################
    ##                      HIP ABDUCTION_LEFT AND RIGHT
    ###############################################################################################
    # ##
    # # # TO CALCULATE ANGLE OF HIP ABDUCTION We need to get the angle between the left shoulder, hip, and knee points. 
    left_hip_abduction_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_hip_abduction_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # # Get the angle between the elbow, shoulder and hip points. 
    # left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # # Get the angle between the right hip, shoulder and elbow points. 
    # right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    # #
    # #
    
    if left_hip_abduction_angle <270 and left_hip_abduction_angle >180 and right_hip_abduction_angle > 75 and right_hip_abduction_angle < 95: #and left_shoulder_angle <40 and right_shoulder_angle <40:
            label = 'LEFT_HIP ABDUCTION  - Bending angle:' + str(360 - left_hip_abduction_angle -90)

    if right_hip_abduction_angle > 90 and right_hip_abduction_angle < 180 and left_hip_abduction_angle < 280 and left_hip_abduction_angle >260: #and left_shoulder_angle <40 and right_shoulder_angle <40:
            label = 'RIGHT_HIP ABDUCTION  - Bending angle:' + str(right_hip_abduction_angle-90)



    #    ##
    # x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    # x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    # mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    # x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    # x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    # dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    # x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
    # lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))

    ####
    # ####
    # #### ER NEUTRAL POSITION:
    # ####
    # ####
    # ####
    # SHOULDER_ER_NEUTRALPOSITION_LEFT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # SHOULDER_ER_NEUTRALPOSITION_RIGHT_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    # # Get the angle between the left shoulder, elbow and wrist points. 
    # left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # # Get the angle between the right shoulder, elbow and wrist points. 
    # right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   

    # #
    # if SHOULDER_ER_NEUTRALPOSITION_LEFT_angle > 20 and left_elbow_angle > 80 and left_elbow_angle < 100:
    #     label = 'LEFT_ER IN NEUTRAL  - Bending angle:' + str(SHOULDER_ER_NEUTRALPOSITION_LEFT_angle)

    # if SHOULDER_ER_NEUTRALPOSITION_RIGHT_angle >20 and right_elbow_angle > 80 and right_elbow_angle < 100:
    #     label = 'RIGHT _ER IN NEUTRAL  - Bending angle:' + str(SHOULDER_ER_NEUTRALPOSITION_RIGHT_angle)

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "h"
def classifyPose_hipAROM(landmarks, output_image, display=False):
    
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left hip, knee and ankle points. 
    # left_hip_knee_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    #                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # # # Get the angle between the right hip, knee and ankle points 
    # right_hip_knee_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    #                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    ## 
    ##
    ###
    ### # Get the angles between the hip, knee and shoulder points 
    ###
    ###
    left_shoulder_hip_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    right_shoulder_hip_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
   
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_point_SHOULDERS_x = (x1+x2)/2
    mid_point_SHOULDERS_y = (y1+y2)/2
    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_point_HIP_y = (y5+y6)/2
    ## For someone to be considered in lying position the avg of the Y cordinate of ANKLES and avg value of y cordinate of SHOULDERS should be close to each other (by lets say 10%)
    x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    mid_point_HANDS_x = (x3+x4)/2
    
    ## For us to distinguish between someone lying facing up or facing down, the algorithm we devised relies on the assumption that the avg value of x cordiante of person WRISTS is > average value of x cordinate of SHOULDERS while lyhing facing up (and vice versa when facing downwards)
    ####
    ####
    ####
    #### Patient should lie with their left side facing the camera
    ####
    ####
    if right_shoulder_hip_knee_angle <165 and left_shoulder_hip_knee_angle >165 and left_shoulder_hip_knee_angle < 190 and mid_point_HANDS_x > mid_point_SHOULDERS_x: #and mid_point_ANKLES_y/mid_point_SHOULDERS_y > 0.5 and mid_point_ANKLES_y/mid_point_SHOULDERS_y < 1.2:
        label = 'LEFT HIP FLEXION  - Bending angle:' + str(180 - right_shoulder_hip_knee_angle)
    if left_shoulder_hip_knee_angle <165 and right_shoulder_hip_knee_angle >165 and right_shoulder_hip_knee_angle < 190 and mid_point_HANDS_x > mid_point_SHOULDERS_x: # and mid_point_ANKLES_y/mid_point_SHOULDERS_y > 0.8 and mid_point_ANKLES_y/mid_point_SHOULDERS_y < 1.2:
        label = 'RIGHT HIP FLEXION  - Bending angle:' + str(180 - left_shoulder_hip_knee_angle)
    ##
    ##
    if right_shoulder_hip_knee_angle > 130 and right_shoulder_hip_knee_angle <180 and mid_point_HANDS_x < mid_point_SHOULDERS_x:# and mid_point_ANKLES_y/mid_point_SHOULDERS_y > 0.8 and mid_point_ANKLES_y/mid_point_SHOULDERS_y < 1.2:
        label = 'RIGHT HIP EXTENSION  - Bending angle:' + str(180 - right_shoulder_hip_knee_angle)
    if left_shoulder_hip_knee_angle > 130 and left_shoulder_hip_knee_angle <180 and mid_point_HANDS_x < mid_point_SHOULDERS_x: # and mid_point_ANKLES_y/mid_point_SHOULDERS_y > 0.8 and mid_point_ANKLES_y/mid_point_SHOULDERS_y < 1.2:
        label = 'LEFT HIP EXTENSION  - Bending angle:' + str(180 - left_shoulder_hip_knee_angle)

    # #########################################################################################################
    # ## Ques: How to distingush IR from ER HIP AROM???? We have to find  way to classify the pose of body when performing IR vs ER and be able to distinguish on the basis of this logis. As the angle that we will be measuring in both cases is the same right/left_InternalRotation_angle, we need only change the logic of output angle to be subtracted by 90 or not, on the basis of the rule/logic we devise for distinguishing between ER and IR  HIP AROM test.
    ## Ans: We will create a signature of angles between certain limb nodes that categgorize the pose of patient when the PT is conducting Hip IR and ER 
    ####
    ####
    ##################################################################################################
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "i"
def classifyPose_hipIR_RotationAROM(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------

    #####################################################################################################
    ##              HIP AROM Internal Rotation
    ##
    right_hip_InternalRotation__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_hip_InternalRotation__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    if right_hip_InternalRotation__angle >99 and right_hip_InternalRotation__angle < 180:
        label = 'RIGHT HIP INTERNAL ROTATION  - Bending angle:' + str(right_hip_InternalRotation__angle - 90)
    if left_hip_InternalRotation__angle >180 and left_hip_InternalRotation__angle < 270:
        label = 'LEFT HIP INTERNAL ROTATION  - Bending angle:' + str(360 - left_hip_InternalRotation__angle - 90)

    ########################################################################################################
    ##          HIP AROM External Rotation
    ##
    # ##
    # if right_hip_InternalRotation__angle < 85: # and left_hip_InternalRotation__angle >80 and left_hip_InternalRotation__angle < 120:
    #     label = 'RIGHT HIP EXTERNAL ROTATION  - Bending angle:' + str(right_hip_InternalRotation__angle)
    # if left_hip_InternalRotation__angle > 275: #and right_hip_InternalRotation__angle >80 and right_hip_InternalRotation__angle < 120:
    #     label = 'LEFT HIP EXTERNAL ROTATION  - Bending angle:' + str(360 - left_hip_InternalRotation__angle)
    # ##
    ##    
    #########################################################################################################
    ## Ques: How to distingush IR from ER HIP AROM???? We have to find  way to classify the pose of body when performing IR vs ER and be able to distinguish on the basis of this logis. As the angle that we will be measuring in both cases is the same right/left_InternalRotation_angle, we need only change the logic of output angle to be subtracted by 90 or not, on the basis of the rule/logic we devise for distinguishing between ER and IR  HIP AROM test.
    # Ans: We will create a signature of angles between certain limb nodes that categgorize the pose of patient when the PT is conducting Hip IR and ER 
    ###
    ###
    ##################################################################################################
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
# "e"
def classifyPose_hipER_RotationAROM(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------

    #####################################################################################################
    ##              HIP AROM Internal Rotation
    ##
    right_hip_InternalRotation__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_hip_InternalRotation__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    # if right_hip_InternalRotation__angle >99 and right_hip_InternalRotation__angle < 180:
    #     label = 'RIGHT HIP INTERNAL ROTATION  - Bending angle:' + str(right_hip_InternalRotation__angle - 90)
    # if left_hip_InternalRotation__angle >180 and left_hip_InternalRotation__angle < 270:
    #     label = 'RIGHT HIP INTERNAL ROTATION  - Bending angle:' + str(360 - left_hip_InternalRotation__angle - 90)

    ########################################################################################################
    ##          HIP AROM External Rotation
    ##
    ##
    if right_hip_InternalRotation__angle < 85: # and left_hip_InternalRotation__angle >80 and left_hip_InternalRotation__angle < 120:
        label = 'LEFT HIP EXTERNAL ROTATION  - Bending angle:' + str(90 - right_hip_InternalRotation__angle)
    if left_hip_InternalRotation__angle > 275: #and right_hip_InternalRotation__angle >80 and right_hip_InternalRotation__angle < 120:
        label = 'RIGHT HIP EXTERNAL ROTATION  - Bending angle:' + str(left_hip_InternalRotation__angle - 270)
    ##
    ##    
    #########################################################################################################
    ## Ques: How to distingush IR from ER HIP AROM???? We have to find  way to classify the pose of body when performing IR vs ER and be able to distinguish on the basis of this logis. As the angle that we will be measuring in both cases is the same right/left_InternalRotation_angle, we need only change the logic of output angle to be subtracted by 90 or not, on the basis of the rule/logic we devise for distinguishing between ER and IR  HIP AROM test.
    # Ans: We will create a signature of angles between certain limb nodes that categgorize the pose of patient when the PT is conducting Hip IR and ER 
    ###
    ###
    ##################################################################################################
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

def classifyPose_KneeAROM(landmarks, output_image, display=False):

    
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.
    #----------------------------------------------------------------------------------------------------------------
    #####################################################################################################
    ##              HIP AROM Internal Rotation
    ##
    left_hip_Knee_ankle__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_hip_Knee_ankle__angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    ##
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_point_SHOULDERS_x = (x1+x2)/2
    mid_point_SHOULDERS_y = (y1+y2)/2
    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_point_HIP_y = (y5+y6)/2
    ## For someone to be considered in lying position the avg of the Y cordinate of HIP and avg value of y cordinate of SHOULDERS should be close to each other (by lets say 10%)
    x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    mid_point_HANDS_x = (x3+x4)/2
    ##algorithm for determing if a body is lying down
    # ratio = abs(mid_point_HIP_y) - abs(mid_point_SHOULDERS_y)
    ##
    if left_hip_Knee_ankle__angle >185 and right_hip_Knee_ankle__angle >175 and right_hip_Knee_ankle__angle < 190 and mid_point_HANDS_x < mid_point_SHOULDERS_x and 1.2 > mid_point_SHOULDERS_y / mid_point_HIP_y > 0.8:
        label = 'LEFT KNEE FLEXION - Bending angle:' + str(360-left_hip_Knee_ankle__angle)
    if right_hip_Knee_ankle__angle >185 and left_hip_Knee_ankle__angle >175 and left_hip_Knee_ankle__angle < 190 and mid_point_HANDS_x < mid_point_SHOULDERS_x and 1.2 > mid_point_SHOULDERS_y / mid_point_HIP_y > 0.8: 
        label = 'RIGHT KNEE FLEXION  - Bending angle:' + str(360-right_hip_Knee_ankle__angle)
    ##
    if left_hip_Knee_ankle__angle >185 and mid_point_HANDS_x > mid_point_SHOULDERS_x and mid_point_HANDS_x < mid_point_SHOULDERS_x and 1.2 > mid_point_SHOULDERS_y / mid_point_HIP_y > 0.8: 
        label = 'LEFT KNEE EXTENSION - Bending angle:' + str(180-left_hip_Knee_ankle__angle)
    if right_hip_Knee_ankle__angle >185 and mid_point_HANDS_x > mid_point_SHOULDERS_x and mid_point_HANDS_x < mid_point_SHOULDERS_x and 1.2 > mid_point_SHOULDERS_y / mid_point_HIP_y > 0.8: 
        label = 'RIGHT KNEE EXTENSION  - Bending angle:' + str(180-right_hip_Knee_ankle__angle)
    ##    
    #########################################################################################################
    ###
    ###
    ##################################################################################################
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

def classifyPose_ElbowAROM(landmarks, output_image, display=False):
    

    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.
    #----------------------------------------------------------------------------------------------------------------
    #####################################################################################################
    ##
     #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
       ##                        
    ###
    ## ##
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] 
    x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_HIP_x,mid_HIP_y,mid_HIP_z = (x3+x4)/2 , (y3+y4)/2, (z3+z4)/2

    GROUND_HIP_NOSE_angle = calculateAngle((mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
                                      (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                      landmarks[mp_pose.PoseLandmark.NOSE.value])
    
    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
    lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))

    #########################################################################################################
    ###
    
        # Check if one leg is straight
    if left_elbow_angle > 230 and dist_between_shoulders < lenght_of_body/6: # and right_elbow_angle > 165:
        # Specify the label of the pose that is tree pose.
        label = 'LEFT ELBOW FLEXION MOTION - Bending angle:'+ str(360 - left_elbow_angle- 180)
        print(label)
        print (right_elbow_angle)

    if right_elbow_angle < 125 and dist_between_shoulders < lenght_of_body/6: # and left_elbow_angle > 165:
        # Specify the label of the pose that is tree pose.
        label = 'RIGHT ELBOW FLEXION MOTION - Bending angle:'+ str(180 - right_elbow_angle)
        print(label)
        print (left_elbow_angle)

    if left_elbow_angle < 190 and left_elbow_angle > 165 and dist_between_shoulders < lenght_of_body/6:
        # Specify the label of the pose that is tree pose.
        label = 'LEFT ELBOW EXTENSION - Bending angle:'+ str(180 - left_elbow_angle)
        print(label)
        print (left_elbow_angle)

    if right_elbow_angle < 190 and right_elbow_angle > 165 and dist_between_shoulders < lenght_of_body/6:
        # Specify the label of the pose that is tree pose.
        label = 'RIGHT ELBOW EXTENSION - Bending angle:'+ str(180 - right_elbow_angle)
        print(label)
        print (right_elbow_angle)

    # if  and dist_between_shoulders > lenght_of_body/5:
    #     # Specify the label of the pose that is tree pose.
    #     label = 'RIGHT ELBOW EXTENSION - Bending angle:'+ str(180 - right_elbow_angle)
        

    ##################################################################################################
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    


    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

#######################################################################################
############################## OAK - D Code goes here##################################
#######################################################################################
#######################################################################################
#####################################################################################
############################### Genes Code was above this ^^^^ #########################
#####################################################################################
pipeline = dai.Pipeline()
# device = dai.Device()
# fps = 30
## Stereo Depth

# Define sources and outputs
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

# Properties
monoLeft = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
# monoLeft.setFps(fps)

monoLeft.out.link(stereo.left)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# monoRight.setFps(fps)

monoRight.out.link(stereo.right)

xDisp = pipeline.createXLinkOut()
xDisp.setStreamName("disparity")
stereo.disparity.link(xDisp.input)

xDepth = pipeline.createXLinkOut()
xDepth.setStreamName("depth")
stereo.depth.link(xDepth.input)



## RGB CAM
# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
##
# downscaleColor = True
# if downscaleColor: camRgb.setIspScale(2, 3)
# # For now, RGB needs fixed focus to properly align with depth.
# # This value was used during calibration
# try:
#     calibData = device.readCalibration2()
#     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
#     if lensPosition:
#         camRgb.initialControl.setManualFocus(lensPosition)
# except:
#     raise

##
# Properties
camRgb.setPreviewSize(640,400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
# Linking
camRgb.preview.link(xoutRgb.input)
# Allignment of RBG and Depth
stereo.setLeftRightCheck(True)
# stereo.setOutputSize(640,400)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(640*2,400*2)


#####################################################################################
#######################################################################################
#######################################################################################
########################################################################################
# Initialize the VideoCapture object to read video recived from PT session
# camera_video = cv2.VideoCapture(0)
# # camera_video = cv2.VideoCapture('project_test_front_camera.avi')
# # camera_video_1 = cv2.VideoCapture (0)
# camera_video.set(3,1280)
# camera_video.set(4,960)
# camera_video_1.set(3,1280)
# camera_video_1.set(4,960)

## Ask for category of PT AROM analysis to be performed 
print("Select the ROM of interest: Press 'l' for LUMBAR AROM, 'a' for SHOULDER AROM, 'c' for CERVICAL AROM,'h' for HIP AROM, t' for top view HIP Abduction AROM, 'i' or 'r' for HIP IR/ER, 'k' for KNEE ROM, 'e' for Elbow ROMS, 'w' for Wrist AROM" )
# a = keyboard.read_key()
a = 'l'
print( "Frame capture Initialized")
    
# with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
with dai.Device(pipeline) as device:    # device.startPipeline(pipeline)
    
    ### Initializing 3D plot figure
    ### Print available device names
    print('Connected cameras:', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    # if device.getBootloaderVersion() is not None:
    #     # print('Bootloader version:', device.getBootloaderVersion())
    # # Device name
    print('Device name:', device.getDeviceInfo())
    ################
    ## RGB Camera
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    ################  
    ## Stereo Camera 
    qDisp = device.getOutputQueue(name="disparity", maxSize=1, blocking=True)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

    # calib = device.readCalibration()
    # baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    # intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, monoRight.getResolutionSize())
    # focalLength = intrinsics[0][0]
    # disp_levels = stereo.initialConfig.getMaxDisparity() / 95
    # dispScaleFactor = baseline * focalLength * disp_levels

    while True: #camera_video.isOpened(): # and camera_video_1.isOpened():
        ####
        ####
        #### Stereo Camera code
        dispFrame = np.array(qDisp.get().getFrame())
        # with np.errstate(divide='ignore'):
        #     calcedDepth = (dispScaleFactor / dispFrame).astype(np.uint16)
        depthFrame = np.array(qDepth.get().getFrame()) 
        # cv2.imshow("depth frame output", depthFrame)
        w,h = depthFrame.shape
        # print(depthFrame.shape)
        # # Note: SSIM calculation is quite slow.
        # ssim_noise = ssim(depthFrame, calcedDepth)
        # print(f'Similarity: {ssim_noise}')
        
        ## RGB Camera code
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        frameRgb = inRgb.getCvFrame()
        # Retrieve 'bgr' (opencv format) frame
        # cv2.imshow("rgb", frameRgb)
        ####
        ##
        # # Read a frame.
        if frameRgb is not None:
            frame = frameRgb
        # ok, frame = camera_video.read()
        #     # print(frameRgb)
        height, width, layers = frame.shape
        size = (width,height)
        ## Check if frame is not read properly.
        #if not ok:
            #continue
        # frame_height, frame_width, _ =  frame.shape
        #     # Resize the frame while keeping the aspect ratio.
        #     # frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        frame_final = frame
            #################################################
            #################################################
            # if landmarks:
                # Perform the Pose Classification.
        if a == 'l': #Lumbar AROM
            # # Perform Pose landmark detection.
            frame, landmarks, landmarks_world = detectPose(frame, pose_video, display=False)

            # x1,y1,z1 = landmarks[mp_pose.PoseLandmark.NOSE.value]
            # # print(x1,y1,z1)
            # print("Printing NOSE DEPTH VALUE (in mm)", depthFrame[int(x1),int(y1)])

            cv2.imshow("depth Frame", depthFrame)
            # print(depthFrame[int(w/2),int(h/2)])

            landmarks_revised = []
            # landmarks = list(landmarks)
            # landmarks_world = list(landmarks_world)

            if len(landmarks) == len(landmarks_world):
                # print("landmarks and world_landmarks are SAME LENGTH")  
        
                for i in range(len(landmarks)):
                    # print(landmarks_world[i])
                    x = landmarks[i][0]
                    y = landmarks[i][1]
                    # print(x,y)
                    try:
                        z = depthFrame[int(x),int(y)]
                    except:
                        pass
                    x_wl = landmarks_world[i][0]
                    y_wl = landmarks_world[i][1]
                    # print(z)
                    # landmarks_world[i][2] = z
                    tuplerevised = (x_wl,y_wl,z)
                    # print(tuplerevised)
                    landmarks_revised.append(tuplerevised)
                    # print(landmarks_revised)
                                
                # for i,j in zip(landmarks,landmarks_world):
                #     x = i[0]
                #     y = i[1]
                #     z = depthFrame[int(x),int(y)]
                #     j[2] = z

      

            try:
                print('Printing coordinates from the revised landmarks list')
                print(landmarks_revised[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                print(landmarks_revised[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                print(landmarks_revised[mp_pose.PoseLandmark.LEFT_WRIST.value])

                arm_angle = calculateAngle_relative3D(landmarks_revised[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks_revised[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks_revised[mp_pose.PoseLandmark.LEFT_WRIST.value])
                print(arm_angle)
            except:
                pass
            # x1,y1,z1 = landmarks_revised[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
            # x2,y2,z2 = landmarks_revised[mp_pose.PoseLandmark.LEFT_HIP.value]
            # x3,y3,z3 = landmarks_revised[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
            # print(x1,y1,z1)
            # print(x2,y2,z2)
            # print(x3,y3,z3)

            # GROUND_HIP_SHOULDER_angle = calculateAngle_relative3D(
            #                                 (x1,y1,z1),
            #                                 (x2,y2,z2),
            #                                 (x3,y3,z3))
     
            # if GROUND_HIP_SHOULDER_angle < 160: #and dist_between_shoulders < lenght_of_body/6: # and left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
            #         # Specify the label of the pose that is tree pose.
            #         print('LUMBAR AROM FORWARD BENDING MOTION - Bending angle:'+ str(GROUND_HIP_SHOULDER_angle))
            # #   
            # else:
            #     print(GROUND_HIP_SHOULDER_angle)  
    
            # frame_final, label = classifyPose_lumbarAROM(landmarks, frame, display=False)
            # frame_final, label = classifyPose_lumbarAROM_3D(landmarks_revised, frame, display=False)

        if a == 'a': #Shoulder AROM
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_shoulderAROM(landmarks, frame, display=False)

        if a == 'c': #Cervical AROM 
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_cervicalAROM(landmarks, frame, display=False)
                
        if a == 't': #TOP VIEW_FOR HIP AROM ABDUCTION
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_hipAROM_topView(landmarks, frame, display=False)
                           
        if a == 'h': #TOP VIEW_FOR SHOULDER ER IN NEUTRAL
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_hipAROM(landmarks, frame, display=False)
                
        if a == 'i': #HIP IR 
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_hipIR_RotationAROM(landmarks, frame, display=False)
                
        if a == 'r': #HIP ER 
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_hipER_RotationAROM(landmarks, frame, display=False)
                
        if a == 'k': # Knee Range of Motion 
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_KneeAROM(landmarks, frame, display=False)
                
        if a == 'e': # Elbow Range of Motion 
            
            frame, landmarks = detectPose(frame, pose_video, display=False)
            frame_final, label = classifyPose_ElbowAROM(landmarks, frame, display=False)
            # # holistic = mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4)
            results_holistic = holistic.process(frame_final)
            mp_drawing.draw_landmarks(
            frame_final, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
            mp_drawing.draw_landmarks(
            frame_final, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if a == 'w': # Wrist Range of Motion 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.4)
            results_holistic = holistic.process(frame)
            label = 'Unknown Pose'
            # mp_drawing.draw_landmarks(
            # frame_final,
            # results_holistic.face_landmarks,
            # mp_holistic.FACEMESH_CONTOURS,
            # landmark_drawing_spec=None,
            # connection_drawing_spec=mp_drawing_styles
            # .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
            frame_final,
            results_holistic.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
       
            mp_drawing.draw_landmarks(
            frame_final, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
            mp_drawing.draw_landmarks(
            frame_final, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if results_holistic.right_hand_landmarks and results_holistic.pose_landmarks:
                if results_holistic.right_hand_landmarks.landmark[20]:
                    ## Right Elbow
                    x1 = results_holistic.pose_landmarks.landmark[14].x * width
                    y1 = results_holistic.pose_landmarks.landmark[14].y * height  
                    ## Right Wrist
                    x2 = results_holistic.right_hand_landmarks.landmark[0].x * width              
                    y2 = results_holistic.right_hand_landmarks.landmark[0].y * height
                    # x2 = results_holistic.pose_landmarks.landmark[16].x * width              
                    # y2 = results_holistic.pose_landmarks.landmark[16].y * height
                    ##Right Middle Finger tip
                    x3 = results_holistic.right_hand_landmarks.landmark[20].x * width                
                    y3 = results_holistic.right_hand_landmarks.landmark[20].y * height

                    right_hand_angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

                    if right_hand_angle < 0:
                        # Add 360 to the found angle.
                        right_hand_angle += 360
                    # print(right_hand_angle)

                    if right_hand_angle < 165:
                        label = 'RIGHT WRIST EXTENSION - Bending angle:'+ str(180 - right_hand_angle)

                    if right_hand_angle > 195:
                        label = 'RIGHT WRIST FLEXION - Bending angle:' + str(180 - right_hand_angle)
            # else:
            #     label = ("right Hand not in frame")
                # continue
                
            if results_holistic.left_hand_landmarks and results_holistic.pose_landmarks:
                if results_holistic.left_hand_landmarks.landmark[20]:
                    # print(results_holistic.left_hand_landmarks.landmark[20].visibility)
                    ## Right Elbow
                    x1 = results_holistic.pose_landmarks.landmark[13].x * width
                    y1 = results_holistic.pose_landmarks.landmark[13].y * height  
                    # ## Right Wrist
                    x2 = results_holistic.left_hand_landmarks.landmark[0].x * width              
                    y2 = results_holistic.left_hand_landmarks.landmark[0].y * height
                    # x2 = results_holistic.pose_landmarks.landmark[15].x * width              
                    # y2 = results_holistic.pose_landmarks.landmark[15].y * height
                    ##Right Middle Finger tip
                    x3 = results_holistic.left_hand_landmarks.landmark[20].x * width                
                    y3 = results_holistic.left_hand_landmarks.landmark[20].y * height
                
                    left_hand_angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
 
                    # Check if the angle is less than zero.
                    if left_hand_angle < 0:
                        # Add 360 to the found angle.
                        left_hand_angle += 360
                    # print(left_hand_angle)
                
                    if left_hand_angle < 165:
                        label = 'LEFT WRIST FLEXION - Bending angle:'+ str(180 - left_hand_angle)

                    if left_hand_angle > 195:
                        label = 'LEFT WRIST EXTENSION - Bending angle:' + str(left_hand_angle - 180)
            # else:
            #     label = ("Left Hand not in frame")
                # continue
            
            color = (0, 0, 255)
            if label != 'Unknown Pose':
                # Update the color (to green) with which the label will be written on the image.
                color = (0, 255, 0)  
            # Write the label on the output image. 
            cv2.putText(frame_final, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


        if cv2.waitKey(1) & 0xFF==ord('s'):
                # break
            print("Select the ROM category : 'l' for LUMBAR AROM, 'a' for SHOULDER AROM, 'c' for CERVICAL AROM,'h' for HIP AROM Flexion/Extension, t' for top view HIP Abduction AROM, 'i' or 'e' for HIP IR/ER, 'k' for KNEE ROM,'e' for elbow ROM, 'w' for Wrist AROM")            
            #listening to input
            a = keyboard.read_key()
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
                # break
            print(label)            
            #returns the value of the LABEL when q is pressed
#########################################################################################################
        # if not ok:
        #     print("Ignoring empty camera frame.")
        # # If loading a video, use 'break' instead of 'continue'.
        #     continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        # image.flags.writeable = False
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ## Drawing Face mesh output for Mediapipe 
        # results = face_mesh.process(frame)
        # # Draw the face mesh annotations on the image.
        # frame.flags.writeable = True
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         mp_drawing.draw_landmarks(
        #             image=frame_final,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing_styles
        #             .get_default_face_mesh_tesselation_style())
        #         mp_drawing.draw_landmarks(
        #             image=frame_final,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_CONTOURS,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing_styles
        #             .get_default_face_mesh_contours_style())
        #         mp_drawing.draw_landmarks(
        #             image=frame_final,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_IRISES,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing_styles
        #             .get_default_face_mesh_iris_connections_style())
       
        
       
        cv2.imshow('MediaPipe Pose Classification', frame)
            # img_array_camera.append(frame_final)
            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed
        
        # cv2.imshow('TopView', frame_1)
    
        k = cv2.waitKey(1) & 0xFF  
            # Check if 'ESC' is pressed.
        if(k == 27):    
            # Break the loop.
            break
    # Release the VideoCapture object and close the windows.st
    # camera_video.release()
 