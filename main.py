import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy.ma.extras import average
import time
import os.path
from picarx import Picarx  # Import Picarx library

# Import AIY Maker Kit and Coral libraries
try:
    from aiymakerkit import vision
    from aiymakerkit import utils
    from pycoral.utils.dataset import read_label_file
    object_detection_available = True
except ImportError:
    print("Warning: AIY Maker Kit or Coral libraries not available. Running without object detection.")
    object_detection_available = False

# Initialize Picar
px = Picarx()

# Camera top filterout
topFilterOut = 0.5

# mask filter variables
bottomLeftCol = 0
bottomRightCol = 1
topLeftCol = 0.2
topRightCol = 0.8
bottomRow = 1
topRow = 0.4

# Canny edge variables
cannyLowThreshold = 240
cannyHighThreshold = 255

# Hough transform variables
rho = 1  # 1
theta = np.pi / 180
threshold = 20  # 20
minLineLength = 50  # 20
maxLineGap = 100  # 500

# Hough line outlier filter variables
percentile = 0.25
outlierMultiplier = 1.5

# Movement parameters - reduced for gentle and stable movement
speed = 15  # Slow base speed (0-100)
max_angle = 30  # Limited max steering angle
kp = 0.3  # Reduced proportional gain for gentler steering
kd = 0.15  # Reduced derivative gain

# Object detection parameters
object_detection_threshold = 0.4
stop_sign_wait_time = 3.0  # seconds to wait at stop sign
last_stop_time = 0  # to prevent repeated stops at the same sign
min_stop_interval = 5.0  # minimum time between stops (seconds)

# Object reaction distances (object area thresholds)
STOP_THRESHOLD = 5000  # pixel area
SLOW_THRESHOLD = 2000  # pixel area

def path(name):
    """Get path to model files"""
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, 'models', name)

# Model paths
ROAD_SIGN_DETECTION_MODEL = path('efficientdet-lite-fullmodelv2.tflite')
ROAD_SIGN_DETECTION_MODEL_EDGETPU = path('efficientdet-lite-fullmodelv2_edgetpu.tflite')
ROAD_SIGN_DETECTION_LABELS = path('fullmodelv2-labels.txt')

# Initialize object detector if available
detector = None
labels = None
if object_detection_available:
    try:
        # Try EdgeTPU version first, fall back to CPU version
        if os.path.exists(ROAD_SIGN_DETECTION_MODEL_EDGETPU):
            detector = vision.Detector(ROAD_SIGN_DETECTION_MODEL_EDGETPU)
            print("Using EdgeTPU model for faster detection")
        else:
            detector = vision.Detector(ROAD_SIGN_DETECTION_MODEL)
            print("Using CPU model (detection may be slower)")
            
        labels = read_label_file(ROAD_SIGN_DETECTION_LABELS)
        print(f"Object detection initialized with {len(labels)} object classes")
    except Exception as e:
        print(f"Error initializing object detection: {e}")
        object_detection_available = False

def regionSelect(image, show=False):
    mask = np.zeros_like(image)
    mask_color = 255
    rows, cols = image.shape[:2]

    bottom_left = [cols * bottomLeftCol, rows * bottomRow]
    top_left = [cols * topLeftCol, rows * topRow]
    bottom_right = [cols * bottomRightCol, rows * bottomRow]
    top_right = [cols * topRightCol, rows * topRow]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def hough_transform(image):
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []

def lineToPixel(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def filter_outliers(df, col):
    if df.empty:
        return df  # Return empty DataFrame instead of failing
    lowPercentile = df[col].quantile(percentile)
    highPercentile = df[col].quantile(1-percentile)
    IQR = highPercentile - lowPercentile
    lower_bound = lowPercentile - outlierMultiplier * IQR
    upper_bound = highPercentile + outlierMultiplier * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def assignLane(lines, imageHeight, prev_left_lane=None, prev_right_lane=None):
    if lines is None or len(lines) == 0:
        return prev_left_lane, prev_right_lane
        
    dtype_dict = {
        "x1": float, "y1": float, "x2": float, "y2": float,
        "length": float, "slope": float, "intercept": float
    }

    left_lines = pd.DataFrame(columns=dtype_dict.keys()).astype(dtype_dict)
    right_lines = pd.DataFrame(columns=dtype_dict.keys()).astype(dtype_dict)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) < 50:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            new_row = pd.DataFrame([[x1, y1, x2, y2, length, slope, intercept]], columns=dtype_dict.keys())

            if slope < 0:
                left_lines = pd.concat([left_lines, new_row], ignore_index=True)
            else:
                right_lines = pd.concat([right_lines, new_row], ignore_index=True)

    # If one of the lane lines is not detected, use previous detection
    if left_lines.empty or right_lines.empty:
        return prev_left_lane, prev_right_lane

    try:
        f_l = filter_outliers(left_lines, 'intercept')
        f_r = filter_outliers(right_lines, 'intercept')
        filtered_left = filter_outliers(f_l, 'slope')
        filtered_right = filter_outliers(f_r, 'slope')

        # If after filtering, we don't have enough lines, use previous detection
        if len(filtered_left) < 2 or len(filtered_right) < 2:
            return prev_left_lane, prev_right_lane

        left_slope = np.average(filtered_left["slope"], weights=filtered_left["length"])
        left_intercept = np.average(filtered_left["intercept"], weights=filtered_left["length"])
        right_slope = np.average(filtered_right["slope"], weights=filtered_right["length"])
        right_intercept = np.average(filtered_right["intercept"], weights=filtered_right["length"])

        y1 = int(imageHeight)
        y2 = int(y1*0.5)
        left_x1 = int((y1 - left_intercept)/left_slope)
        left_x2 = int((y2 - left_intercept)/left_slope)

        right_x1 = int((y1 - right_intercept)/right_slope)
        right_x2 = int((y2 - right_intercept)/right_slope)

        leftLane = (left_x1, left_x2, y1, y2, left_slope, left_intercept)
        rightLane = (right_x1, right_x2, y1, y2, right_slope, right_intercept)
        return leftLane, rightLane
    except Exception as e:
        print(f"Lane detection error: {e}")
        return prev_left_lane, prev_right_lane

def frameProcess(image, prev_left_lane=None, prev_right_lane=None, show=False):
    h = image.shape[0]

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)
    image_canny = cv2.Canny(image_blurred, cannyLowThreshold, cannyHighThreshold)
    image_processed = regionSelect(image_canny)
    lines = hough_transform(image_processed)

    leftLane, rightLane = assignLane(lines, h, prev_left_lane, prev_right_lane)
    
    if leftLane is None or rightLane is None:
        return None, None, image_canny

    return leftLane, rightLane, image_canny

def calculate_steering(left_lane, right_lane, frame_width):
    """Calculate steering angle based on lane positions"""
    try:
        if left_lane is None or right_lane is None:
            return 0
            
        left_x1, _, _, _, _, _ = left_lane
        right_x1, _, _, _, _, _ = right_lane
        
        # Calculate center point between lanes at the bottom of the image
        center_lane = (left_x1 + right_x1) // 2
        
        # Calculate frame center
        frame_center = frame_width // 2
        
        # Calculate error (how far from center)
        error = center_lane - frame_center
        
        # Calculate steering angle proportional to error
        # Negative angle means turn left, positive means turn right
        steering_angle = (error / (frame_width / 2)) * max_angle
        
        # Limit steering angle
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        
        return steering_angle
    except Exception as e:
        print(f"Steering calculation error: {e}")
        return 0

def detect_objects(frame):
    """Detect objects in the frame using the object detection model"""
    if not object_detection_available or detector is None:
        return []
    
    try:
        objects = detector.get_objects(frame, threshold=object_detection_threshold)
        return objects
    except Exception as e:
        print(f"Object detection error: {e}")
        return []

def process_objects(objects, frame_width, frame_height):
    """Process detected objects and determine actions"""
    global last_stop_time
    
    if not objects:
        return None, None
        
    current_time = time.time()
    
    # Priority objects and their actions
    priority_object = None
    action = None
    max_area = 0
    
    for obj in objects:
        # Calculate object size (area)
        obj_width = obj.bbox.xmax - obj.bbox.xmin
        obj_height = obj.bbox.ymax - obj.bbox.ymin
        obj_area = obj_width * obj_height
        
        # Get object center X position
        obj_center_x = (obj.bbox.xmin + obj.bbox.xmax) / 2
        # Calculate how far object is from center (normalized -1 to 1)
        center_offset = (obj_center_x - (frame_width / 2)) / (frame_width / 2)
        
        label = labels[obj.id]
        
        # If object is larger than current priority and relevant
        if obj_area > max_area:
            if label == "sign_stop" and obj_area > SLOW_THRESHOLD:
                # Only stop if enough time has passed since last stop
                if current_time - last_stop_time > min_stop_interval:
                    if obj_area > STOP_THRESHOLD:
                        priority_object = obj
                        action = "stop"
                        max_area = obj_area
                    else:
                        priority_object = obj
                        action = "slow"
                        max_area = obj_area
                        
            elif label in ["sign_yield"] and obj_area > SLOW_THRESHOLD:
                priority_object = obj
                action = "slow"
                max_area = obj_area
                
            elif label in ["duck_regular", "duck_specialty"] and obj_area > SLOW_THRESHOLD:
                priority_object = obj
                action = "slow"
                max_area = obj_area
                
            elif label in ["sign_oneway_right"] and obj_area > SLOW_THRESHOLD:
                priority_object = obj
                action = "turn_right"
                max_area = obj_area
                
            elif label in ["sign_oneway_left"] and obj_area > SLOW_THRESHOLD:
                priority_object = obj
                action = "turn_left"
                max_area = obj_area
                
            elif label == "sign_noentry" and obj_area > SLOW_THRESHOLD:
                priority_object = obj
                action = "turn_around"
                max_area = obj_area
    
    if priority_object:
        label = labels[priority_object.id]
        print(f"Detected {label}: {action}")
        
        # If stopping at a stop sign, update the last stop time
        if action == "stop" and label == "sign_stop":
            last_stop_time = current_time
    
    return priority_object, action

def draw_objects(frame, objects):
    """Draw bounding boxes and labels for detected objects"""
    if not objects:
        return frame
        
    result = frame.copy()
    for obj in objects:
        # Draw box
        x1, y1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
        x2, y2 = int(obj.bbox.xmax), int(obj.bbox.ymax)
        
        # Color based on object type
        color = (0, 255, 0)  # Default green
        if labels[obj.id].startswith("sign_stop"):
            color = (0, 0, 255)  # Red for stop signs
        elif labels[obj.id].startswith("duck"):
            color = (255, 165, 0)  # Orange for ducks
            
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{labels[obj.id]}: {obj.score:.2f}"
        cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return result

def handle_action(action):
    """Execute specific action based on detected objects"""
    global speed
    
    if action == "stop":
        print("Stopping at stop sign")
        px.stop()
        time.sleep(stop_sign_wait_time)
        px.forward(speed * 0.5)  # Resume at half speed
        time.sleep(1)  # Give time to get past the sign
        return True
        
    elif action == "slow":
        print("Slowing down")
        px.forward(speed * 0.5)
        return True
        
    elif action == "turn_right":
        print("Turning right at one-way sign")
        px.set_dir_servo_angle(max_angle * 0.8)
        px.forward(speed * 0.7)
        time.sleep(1.5)
        px.set_dir_servo_angle(0)
        return True
        
    elif action == "turn_left":
        print("Turning left at one-way sign")
        px.set_dir_servo_angle(-max_angle * 0.8)
        px.forward(speed * 0.7)
        time.sleep(1.5)
        px.set_dir_servo_angle(0)
        return True
        
    elif action == "turn_around":
        print("Turning around at no entry sign")
        px.set_dir_servo_angle(max_angle)
        px.forward(speed * 0.7)
        time.sleep(3)  # Longer time for U-turn
        px.set_dir_servo_angle(0)
        return True
        
    return False

def drive_car():
    """Main function to drive the car based on lane detection and object detection"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Set a slow, safe speed
    global speed
    speed = 15  # Slow speed for safety
    
    # Initialize previous lane detections
    prev_left_lane = None
    prev_right_lane = None
    
    # Initialize error for derivative control
    last_error = 0
    
    # For display overlay
    show_display = True
    
    try:
        print("Starting lane following with object detection...")
        print("Press Ctrl+C to stop")
        
        # Set initial car direction
        px.set_dir_servo_angle(0)
        
        # Wait for camera to initialize fully
        print("Initializing camera...")
        time.sleep(2)
        
        # Start moving forward at slow speed
        print("Starting to move forward...")
        px.forward(speed)
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame for lane detection
            left_lane, right_lane, processed = frameProcess(frame, prev_left_lane, prev_right_lane)
            
            # Update previous lane detections
            if left_lane is not None:
                prev_left_lane = left_lane
            if right_lane is not None:
                prev_right_lane = right_lane
            
            # Detect objects if available
            objects = []
            if object_detection_available:
                objects = detect_objects(frame)
                
                # Display frame with detected objects if show_display is enabled
                if show_display and len(objects) > 0:
                    display_frame = draw_objects(frame, objects)
                    cv2.imshow("Object Detection", display_frame)
                
                # Process objects and determine actions
                _, action = process_objects(objects, frame.shape[1], frame.shape[0])
                
                # Handle object-based actions
                if action:
                    action_handled = handle_action(action)
                    if action_handled:
                        continue  # Skip lane following for this frame as we're handling an object
            
            # Lane following (if no object actions were taken)
            if left_lane is not None and right_lane is not None:
                # Calculate center error
                left_x1, _, y1, _, _, _ = left_lane
                right_x1, _, _, _, _, _ = right_lane
                center_lane = (left_x1 + right_x1) // 2
                frame_center = frame.shape[1] // 2
                error = center_lane - frame_center
                
                # Simple PD controller
                derivative = error - last_error
                steering = kp * error + kd * derivative
                last_error = error
                
                # Limit steering angle for gentle movement
                steering = min(max(steering, -max_angle), max_angle)
                
                # Apply steering - use a reduced max angle for smoother movement
                px.set_dir_servo_angle(steering * 0.8)
                
                # Keep driving at constant slow speed
                px.forward(speed)
                
                if show_display:
                    # Draw lane lines on display
                    display_frame = frame.copy()
                    if left_lane and right_lane:
                        left_x1, left_x2, y1, y2, _, _ = left_lane
                        right_x1, right_x2, _, _, _, _ = right_lane
                        
                        # Draw left lane line
                        cv2.line(display_frame, (left_x1, y1), (left_x2, y2), (255, 0, 0), 3)
                        # Draw right lane line
                        cv2.line(display_frame, (right_x1, y1), (right_x2, y2), (255, 0, 0), 3)
                        # Draw center line
                        center_x1 = (left_x1 + right_x1) // 2
                        center_x2 = (left_x2 + right_x2) // 2
                        cv2.line(display_frame, (center_x1, y1), (center_x2, y2), (0, 255, 0), 2)
                    
                    cv2.putText(display_frame, f"Steering: {steering:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Lane Following", display_frame)
                    
                print(f"Lane following: error={error}, steering={steering:.2f}")
            else:
                # If no lanes detected, maintain straight direction but reduce speed
                px.set_dir_servo_angle(0)
                px.forward(speed * 0.7)  # Even slower when no lanes detected
                print("No lanes detected, maintaining straight direction")
            
            # Key handling for display
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_display = not show_display
                if not show_display:
                    cv2.destroyAllWindows()
            
            # Small delay
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        px.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Car stopped and resources released")

def test_object_detection():
    """Test object detection without moving the car"""
    if not object_detection_available:
        print("Object detection is not available. Please install required libraries.")
        return
        
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        print("Starting object detection test mode...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Detect objects
                objects = detect_objects(frame)
                
                # Draw objects on frame
                display_frame = draw_objects(frame, objects)
                
                # Process objects for actions (don't actually perform them)
                priority_object, action = process_objects(objects, frame.shape[1], frame.shape[0])
                
                # Display info on frame
                if priority_object:
                    label = labels[priority_object.id]
                    cv2.putText(display_frame, f"Action: {action} for {label}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("Object Detection Test", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            
            time.sleep(0.05)
    except Exception as e:
        print(f"Error in test mode: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test mode ended")

if __name__ == "__main__":
    try:
        # Ask user which mode to run
        print("\nPicarX Lane Following with Object Detection")
        print("1: Drive with Lane Following and Object Detection")
        print("2: Test Object Detection (no movement)")
        print("3: Quit")
        
        choice = input("Select mode: ")
        
        if choice == "1":
            drive_car()
        elif choice == "2":
            test_object_detection()
        else:
            print("Exiting program")
    except Exception as e:
        print(f"Fatal error: {e}")
        # Make sure car is stopped in case of error
        try:
            px.stop()
        except:
            pass
