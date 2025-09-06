# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 09:50:28 2023

@author: ahmad
"""

# Import lobraries - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from ultralytics import YOLO
import cv2
import pyrealsense2
import numpy as np
from rplidar import RPLidar
import time
from realsense_camera import *



# Define functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def get_yolo_detection(image_frame): # Completed
    """
    This function will feed the given image to yolo and check for the detection result,
    if there is no detection then will return "is_detection" = False,
    otherwise will return "is_detection" = True, and return "crack_index", and "bb_coordinates".
    
    
    Parameters
    -------
    image_frame (2d-array): BGR frame (BGR not RGB because of the camera (I guess!?)).
    model_path (str): path to the trained YOLO model
    
    
    Returns
    -------
    is_detection (bool): This indicates if there is a detection in the given image or not. 
    crack_index (int): The class index of the detected crack.
    bb_coordinates (tuple): This contains [x-center, y-center, bb-width, bb-height].
    
    """
    
    model_path = r"D:\Downloads\Crack detection code\best.pt" # Give trained YOLO model path (str)
    
    # Initialize YOLOv8 model
    model = YOLO(model_path)
    
    # Run YOLOv8 on the image
    results = model(image_frame, conf=0.4)
    
    # Process YOLOv8 results
    for result in results:
        boxes = result.boxes
        if boxes: # True if at least one crack detected
            is_detection = True
            confs = [box.conf for box in boxes]
            index = confs.index(max(confs))
            
            # Class index
            crack_index = int(boxes[index].cls)
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = boxes[index].xyxy[0]
            print(f"x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}")
            
            # Process the bounding box data
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            bb_coordinates = int(x_center), int(y_center), int(w), int(h)
            
            
        else: # No crack detected
            is_detection = False
            crack_index = 0
            bb_coordinates = 0,0,0,0
            
    return is_detection, crack_index, bb_coordinates



def get_distance_to_crack(bb_coordinates, depth_frame): # Completed
    """
    This function will calculate the horizontal distance to crack by using the "depth_frame",
    and the "bb_coordinates" and return the "distance_to_crack".
    
    
    Parameters
    -------
    bb_coordinates (tuple): This contains [x-center, y-center, bb-width, bb-height].
    depth_frame (2d-array): Depth frame.
    
    
    Returns
    -------
    distance_to_crack (float): The calculated distance to the crack in (meter). 
    
    """
    
    x_center, y_center, _, _ = bb_coordinates
    
    # Get distance to crack in (millimeter)
    distance_to_crack = depth_frame[y_center, x_center] 
    
    # Convert unit to (meter)
    distance_to_crack = distance_to_crack / 1000
    return distance_to_crack
    
    


def is_valid_distance_to_crack(distance_to_crack): # Completed
    """
    This function will check if "distance_to_crack" is within the valid range (i.e. < 1.2 meter).
    If it was in the valid range, then it will return "True", otherwise, it will return False
    
    
    Parameters
    -------
    distance_to_crack (float): The calculated distance to the crack in (meter).
    
    
    Returns
    -------
    is_valid_distance (bool): If "distance_to_crack" < 1.20 meter, then it returns "True", otherwise it returns "False" .
    
    """
    
    REQUIRED_DISTANCE = 3 # meter
    
    if distance_to_crack <= REQUIRED_DISTANCE:
        is_valid_distance = True
    else:
        is_valid_distance = False
    
    return is_valid_distance
    


def get_crack_area_and_diameter(bb_coordinates): # Completed
    """
    This function will calculate the "crack_area" in (mm^2) and "crak_diameter" in (mm) based on "bb_coordinates".

    
    Parameters
    -------
    bb_coordinates (tuple): This contains [x-center, y-center, bb-width, bb-height].
    

    Returns
    -------
    crack_area (float): The area of the crack in (mm^2).
    crack_diameter (float): The diameter of the crack in (mm).
    
    """
    
    
    # Extracting values from bb_coordinates
    x_center, y_center, bb_width, bb_height= bb_coordinates
    
    # Calculating crack area 
    crack_area = bb_width * bb_height
    
    # Calculating crack diameter using the diagonal length of the bounding box
    crack_diameter = np.sqrt(bb_width**2 + bb_height**2)
    
    return crack_area, crack_diameter
    
    

def get_time_to_crack(vehicle_speed, distance_to_crack, latency_time): # Completed
    """
    This function will calculate the required time to reach the crack based on the given parameters.
    
    
    Parameters
    -------
    vehicle_speed (float): speed of the vehicle in (meter/second).
    distance_to_crack (float): The calculated distance to the crack in (meter).
    latency_time (float): latency time

    
    Returns
    -------
    time_to_crack (float): the required time to reach the crack in (second)
    
    """
    
    # Get time to reach crack without accounting for latency
    time_to_crack = distance_to_crack / vehicle_speed # unit is (second)
    print(f"time_to_crack = {distance_to_crack} divided by {vehicle_speed} = {time_to_crack}")
    # Get time to reach crack with accounting for latency
    time_to_crack = time_to_crack - latency_time # unit is (second)
    
    return time_to_crack
    




def get_crack_class(crack_index, bb_coordinates): # Completed
    """
    This function will return the class of the crack based on the "crack_index" and based on the "bb_width" and
    "bb_height". If crack class is "Linear" then it will be converted into either "Longitudinal" or "Transverse".
    
    
    Parameters
    -------
    crack_index (int): The class index of the detected crack.
    bb_coordinates (tuple): This contains [x-center, y-center, bb-width, bb-height].


    Returns
    -------
    crack_class (str): the class of the detecte crcak.
    
    """
    # ORIGINAL_CLASSES = ['Linear-Crack', 'Alligator-Crack', 'pothole']
    NEW_CLASSES = ['Longitudinal', 'Transverse', 'Alligator', 'Pothole']
    
    _, _, bb_width, bb_height = bb_coordinates
    
    if crack_index == 0: # Linear-Crack
        if bb_height > bb_width:
            crack_class = NEW_CLASSES[0] # Set it to "Longitudinal" 
        else:
            crack_class = NEW_CLASSES[1] # Set it to "Transverse" 
    
    elif crack_index == 1: # Alligator-Crack
        crack_class = NEW_CLASSES[2] # Set it to "Alligator" 
        
    else:
        crack_class = NEW_CLASSES[3] # Set it to "Pothole" 
    
    
    return crack_class


def get_crack_severity(crack_class, crack_area, crack_diameter, crack_depth): # Completed
    
    """
    This function will calculate crack severity based on the given paramters.
    the crack severity can be one of the following: ['Low', 'Medium', 'High']

    
    Parameters
    -------
    crack_class (str): the class of the detecte crcak.
    crack_area (float): The area of the crack in (mm^2).
    crack_diameter (float): The diameter of the crack in (mm).
    crack_depth (float): The depth of the crack in (mm).


    Returns
    -------
    crack_severity (str): The severity of the crack: ['Low', 'Medium', 'High'].
    
    """
    
    
    if crack_class == 'Pothole':

        if crack_depth <= 25:
            if crack_diameter <= 450:
                crack_severity = 'Low'
            else:
                crack_severity = 'Medium'
        elif 26 <= crack_depth <= 50:
            if crack_diameter <= 200:
                crack_severity = 'Low'
            elif 201 <= crack_diameter <= 450:
                crack_severity = 'Medium'
            else:
                crack_severity = 'High'
        else:
            if crack_diameter <= 450:
                crack_severity = 'Medium'
            else:
                crack_severity = 'High'


    elif crack_class == 'Alligator':
        # Define A1 and A2 values
        A1 = 30000  # mm^2
        A2 = 60000 # mm^2


        if crack_area < A1:
            crack_severity = 'Low'
        elif A1 <= crack_area <= A2:
            crack_severity = 'Medium'
        else:
            crack_severity = 'High'


    else: # "Longitudinal" or "Transverse"
        if crack_depth < 10:
            crack_severity = 'Low'
        elif 11 <= crack_depth <= 75:
            crack_severity = 'Medium'
        else:
            crack_severity = 'High'


    return crack_severity


def low_pass_filter(distances, angles, window_size=8):
    # Adjust padding to ensure the output length matches the input length
    padding = (window_size - 1) // 2
    smoothed_distances = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
    
    # Adjust angles to match the length of smoothed distances
    angles = angles[padding: -padding]

    return smoothed_distances, angles


def normalize(distances, angles):
    num_weights = len(distances)

    highest_weight = 4
    lowest_weight = 1

    if num_weights % 2 == 0:
        weights1 = np.linspace(highest_weight, lowest_weight, num_weights // 2)
        weights2 = np.linspace(lowest_weight, highest_weight, num_weights // 2)
        weights = np.concatenate((weights1, weights2))
    else:
        weights1 = np.linspace(highest_weight, lowest_weight, (num_weights - 1) // 2)
        weights2 = np.linspace(lowest_weight, highest_weight, (num_weights + 1) // 2)
        weights = np.concatenate((weights1, weights2))

    normalized_distances = [distance - weight * abs(angle - 180) for distance, angle, weight in zip(distances, angles, weights)]

    return normalized_distances

def get_depth(distances_denoised):
    
    first_window_size = 5
    second_window_size = 3
    stride = 2
    threshold = 30  # 30 mm

    depth_values = []
    
    for i in range(0, len(distances_denoised) - first_window_size, stride):
        first_window = distances_denoised[i:i+first_window_size]
        avg_first_window = np.mean(first_window)

        second_window = distances_denoised[i+first_window_size:i+first_window_size+second_window_size]
        avg_second_window = np.mean(second_window)

        depth_difference = avg_second_window - avg_first_window 

        if depth_difference > threshold:
            depth_values.append(depth_difference)
            #print(f"{depth_difference} appended!")

    if depth_values:
        depth = np.mean(depth_values)
    else:
        depth = 0
    
    #print(f"depth = {depth} mm")
    
    return depth

def collect_and_process_lidar_data(runtime_seconds=1):
    # Replace 'COM3' with the port where your RPLIDAR is connected
        PORT_NAME = 'COM3'

        # Create an RPLidar object
        lidar = RPLidar(PORT_NAME)
        lidar.__init__('COM3', 1000000, 6, None)
        depths_to_average = []
        start_time = time.time()
        depth = 0 
        try:
            lidar.start_motor()
            for scan in lidar.iter_scans():
                distances, angles = [], []
                for (_, angle, distance) in scan:
                    # Filter data based on angle range
                    if 150 <= angle <= 210:
                        distances.append(distance)
                        angles.append(angle)
                
                # print(f"This is scan number {n}")
                if distances:
                    # Smooth the signal
                    smoothed_distances, angles = low_pass_filter(distances, angles)

                    # Preprocess the distances based on the angle
                    normalized_distances = normalize(smoothed_distances, angles)

                    # Process the data to get the depth
                    depth = get_depth(normalized_distances)

                    # If depth anomaly is detected (e.g., depth > threshold)
                    if len(depths_to_average) == 2:
                        print("This is the second scan after the detection")
                        depths_to_average.append(depth)

                    elif len(depths_to_average) == 1:
                        print("This is the first scan after the detection")
                        depths_to_average.append(depth)

                    elif depth > 30:
                        print("Depth => 30 mm was detected!")
                        print("Scanning 2 more times...")
                        # Store the depth and the depths of the next two scans
                        depths_to_average = [depth]

                    
                    if len(depths_to_average) == 3:
                        # Calculate the average depth
                        avg_depth = np.mean(depths_to_average)
                        depths_to_average = []
                        return avg_depth
                
                # Check if the runtime duration is reached
                if time.time() - start_time >= runtime_seconds: # if run_time_duration reached and no reading, return 0
                    return avg_depth

        except Exception as e:
            print("Stopping due to: ", str(e))
        finally:
            lidar.stop_motor()
            


rs = RealsenseCamera()

while True:
    i = 1
    # Get frames from the RealSense camera
    ret, image_frame, depth_frame = rs.get_frame_stream()
    time_1 = time.time() # (float)
    GPS_coordinates = np.array([0, 0, 0, 0])  # (list) [Latitude, Longitude]
    vehicle_speed = 1 # m/s
    

    # bb_coordinates (list)==> [bb_x-center, bb_y-center, bb_width, bb_height]
    is_detection, crack_index, bb_coordinates = get_yolo_detection(image_frame)
    print(f"is_detection = {is_detection}")
    print(f"bb_coordinates = {bb_coordinates}")
    
    if not is_detection:
        print("No Detection ==> Skipping to next frame...")
        continue # skip this iteration and start a new iteration of the loop
        
    distance_to_crack = get_distance_to_crack(bb_coordinates, depth_frame)
    print(f"distance_to_crack = {distance_to_crack}")
    
    is_valid_distance = is_valid_distance_to_crack(distance_to_crack)
    
    if not is_valid_distance:
        print("Not a valid distance ==> Skipping to next frame...")
        
        # ________________________________
        
        x_center = bb_coordinates[0]
        y_center = bb_coordinates[1]
        width = bb_coordinates[2]
        height = bb_coordinates[3]
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        distance_str = f"Distance to crack: {distance_to_crack:.2f} meters"

        cv2.circle(image_frame, (x_center, y_center), 5, (0, 255, 0), -1)  # Draw a circle at the center
       
        cv2.putText(image_frame, distance_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Draw text on the image frame

        cv2.rectangle(image_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness
        # Save the image frame with the center marker and distance information
        output_filename = f"crack_{i}.jpg"
        
        # Save the modified image
        cv2.imshow('RealSense Camera', image_frame)
        cv2.waitKey(1)
        time.sleep(5)
        cv2.destroyAllWindows()

        # ________________________________
        
        
        continue # skip this iteration and start a new iteration of the loop
    
    time_2 = time.time()
    latency_time = time_2 - time_1
    print(f"latency_time = {latency_time}")
    time_to_crack = get_time_to_crack(vehicle_speed, distance_to_crack, latency_time) 
    print(f"time_to_crack = {time_to_crack}")
    if time_to_crack <= 0:
        print("OMG! Crack have been skipped!! ==> skipping to next fram...")
        
        # ________________________________
        
        x_center = bb_coordinates[0]
        y_center = bb_coordinates[1]
        width = bb_coordinates[2]
        height = bb_coordinates[3]
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        distance_str = f"Distance to crack: {distance_to_crack:.2f} meters"

        cv2.circle(image_frame, (x_center, y_center), 5, (0, 255, 0), -1)  # Draw a circle at the center
       
        cv2.putText(image_frame, distance_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Draw text on the image frame

        cv2.rectangle(image_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness
        # Save the image frame with the center marker and distance information
        output_filename = f"crack_{i}.jpg"
        
        # Save the modified image
        cv2.imshow('RealSense Camera', image_frame)
        cv2.waitKey(1)
        time.sleep(5)
        cv2.destroyAllWindows()

        # ________________________________
        
        
        continue
    
    print(f"time_to_crack = {time_to_crack}\nTaking lidar reading after {time_to_crack} seconds...")
    time.sleep(time_to_crack)
    
    
    crack_depth = collect_and_process_lidar_data()
    print(f"crack_depth = {crack_depth} [mm]")
    
    
    crack_area, crack_diameter = get_crack_area_and_diameter(bb_coordinates)
    print(f"crack_area = {crack_area}")
    print(f"crack_diameter = {crack_diameter}")

    crack_class = get_crack_class(crack_index, bb_coordinates)
    print(f"crack_class = {crack_class}")
    
    crack_severity = get_crack_severity(crack_class, crack_area, crack_diameter, crack_depth)
    print(f"crack_severity = {crack_severity}")
    
    
    # Saving the image of the crack with the bounding box and center of bounding box and distance to crack and class of crack
    x_center = bb_coordinates[0]
    y_center = bb_coordinates[1]
    width = bb_coordinates[2]
    height = bb_coordinates[3]
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    distance_str = f"Distance to {crack_class}: {distance_to_crack:.2f} meters"

    cv2.circle(image_frame, (x_center, y_center), 5, (0, 255, 0), -1)  # Draw a circle at the center
   
    cv2.putText(image_frame, distance_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Draw text on the image frame

    cv2.rectangle(image_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness
    # Save the image frame with the center marker and distance information
    output_filename = f"{crack_class}_{i}.jpg"
    
    # Save the modified image
    cv2.imwrite(output_filename, image_frame)
    
    print(f"Image saved as {output_filename}")
    
    i+=1
    
    # Wait 5 second before starting new iteration
    print(20*"*")
    time.sleep(5)
        
    
