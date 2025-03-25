import numpy as np
import cv2
import time
import pandas as pd
import os
import sys

try:
    from picarx import Picarx  # Import Picarx library
except ImportError:
    print("Error: Cannot import Picarx library.")
    sys.exit(1)

# Disable OpenCV GUI to prevent display errors
os.environ["OPENCV_AVOID_GUI"] = "1"

# Initialize Picar
try:
    px = Picarx()
except Exception as e:
    print("Error initializing Picarx: {}".format(e))
    sys.exit(1)

# Camera filter parameters
bottomLeftCol = 0
bottomRightCol = 1
topLeftCol = 0.15
topRightCol = 0.85
bottomRow = 1
topRow = 0.5

# Canny edge variables
cannyLowThreshold = 50
cannyHighThreshold = 150

# Hough transform variables
rho = 1
theta = np.pi / 180
threshold = 15
minLineLength = 20
maxLineGap = 200

# Hough line outlier filter variables
percentile = 0.2
outlierMultiplier = 1.8

# Movement parameters - REDUCED SPEED
speed = 8  # Base speed
max_angle = 30  # Max steering angle
kp = 0.4  # Increased for more responsive steering
kd = 0.15  # Derivative gain

# White line detection parameters
white_threshold = 200  # Threshold for white color detection

# Grayscale sensor parameters for solid white line detection
GRAYSCALE_THRESHOLD = 60  # Threshold for detecting solid white line
STOP_DURATION = 1.0  # Time to stop when white line is detected
MIN_TIME_BETWEEN_STOPS = 3.0  # Minimum time between stops

# Last time we stopped for a line
last_stop_time = 0

def check_solid_white_line():
    """Check grayscale sensors for solid white line crossing"""
    adc_value_list = px.get_grayscale_data()
    
    if adc_value_list:
        left_value = adc_value_list[0]
        middle_value = adc_value_list[1]
        right_value = adc_value_list[2]
        
        print("Grayscale sensors - L:{}, M:{}, R:{}".format(
            left_value, middle_value, right_value))
        
        # Lower values = whiter surface
        if (middle_value < GRAYSCALE_THRESHOLD) or \
           (left_value < GRAYSCALE_THRESHOLD and right_value < GRAYSCALE_THRESHOLD):
            print("Solid white line detected by grayscale sensors!")
            return True
            
    return False

def frameProcess(image, prev_left_lane=None, prev_right_lane=None):
    """Process frame for lane detection - simplified for better performance"""
    h, w = image.shape[:2]
    
    # Convert to grayscale and apply blur
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # Apply threshold for better line detection
    _, binary = cv2.threshold(image_blurred, 170, 255, cv2.THRESH_BINARY_INV)
    
    # Apply Canny edge detection
    image_canny = cv2.Canny(binary, cannyLowThreshold, cannyHighThreshold)
    
    # Apply region mask
    mask = np.zeros_like(image_canny)
    rows, cols = image_canny.shape[:2]
    
    # Define region of interest
    bottom_left = [cols * bottomLeftCol, rows * bottomRow]
    top_left = [cols * topLeftCol, rows * topRow]
    bottom_right = [cols * bottomRightCol, rows * bottomRow]
    top_right = [cols * topRightCol, rows * topRow]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    # Apply mask
    image_processed = cv2.bitwise_and(image_canny, mask)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(image_processed, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # Process lines to find left and right lanes
    left_lane = None
    right_lane = None
    
    if lines is not None and len(lines) > 0:
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip nearly horizontal or vertical lines
                if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope
                if -0.9 < slope < -0.1:  # Left lane (negative slope)
                    left_lines.append(line)
                elif 0.1 < slope < 0.9:  # Right lane (positive slope)
                    right_lines.append(line)
        
        # Process left lane
        if left_lines:
            x1_sum, y1_sum, x2_sum, y2_sum = 0, 0, 0, 0
            for line in left_lines:
                for x1, y1, x2, y2 in line:
                    x1_sum += x1
                    y1_sum += y1
                    x2_sum += x2
                    y2_sum += y2
            
            x1 = int(x1_sum / len(left_lines))
            y1 = int(y1_sum / len(left_lines))
            x2 = int(x2_sum / len(left_lines))
            y2 = int(y2_sum / len(left_lines))
            
            left_lane = (x1, x2, y1, y2)
        elif prev_left_lane is not None:
            left_lane = prev_left_lane
            
        # Process right lane
        if right_lines:
            x1_sum, y1_sum, x2_sum, y2_sum = 0, 0, 0, 0
            for line in right_lines:
                for x1, y1, x2, y2 in line:
                    x1_sum += x1
                    y1_sum += y1
                    x2_sum += x2
                    y2_sum += y2
            
            x1 = int(x1_sum / len(right_lines))
            y1 = int(y1_sum / len(right_lines))
            x2 = int(x2_sum / len(right_lines))
            y2 = int(y2_sum / len(right_lines))
            
            right_lane = (x1, x2, y1, y2)
        elif prev_right_lane is not None:
            right_lane = prev_right_lane
    
    return left_lane, right_lane, image_processed

def calculate_steering(left_lane, right_lane, frame_width):
    """Calculate steering angle based on lane positions"""
    try:
        frame_center = frame_width // 2
        
        if left_lane is not None and right_lane is not None:
            # We have both lanes - stay in the middle
            left_x1, _, _, _ = left_lane
            right_x1, _, _, _ = right_lane
            
            # Calculate center point between lanes at the bottom of the image
            center_lane = (left_x1 + right_x1) // 2
            
            # Calculate error (how far from center)
            error = center_lane - frame_center
            
        elif right_lane is not None:
            # We only have right lane - maintain fixed distance
            right_x1, _, _, _ = right_lane
            
            # Target position is 100 pixels to the left of right lane
            target_position = right_x1 - 100
            
            # Calculate error
            error = target_position - frame_center
            
        elif left_lane is not None:
            # We only have left lane - maintain fixed distance
            left_x1, _, _, _ = left_lane
            
            # Target position is 120 pixels to the right of left lane
            target_position = left_x1 + 120
            
            # Calculate error
            error = target_position - frame_center
            
        else:
            # No lanes detected
            return 0
        
        # Calculate steering angle
        steering_angle = (error / (frame_width / 2)) * max_angle
        
        # Limit steering angle
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        
        return steering_angle
    except Exception as e:
        print("Steering calculation error: {}".format(e))
        return 0

def drive_car():
    """Main function to drive the car based on lane detection"""
    # Initialize camera
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
            
        # Lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    except Exception as e:
        print("Error setting up camera: {}".format(e))
        return
    
    # Initialize previous lane detections
    prev_left_lane = None
    prev_right_lane = None
    
    # Initialize error for derivative control
    last_error = 0
    
    # For smoothing steering angles
    steering_history = [0] * 3
    
    # Keep track of last stop time
    global last_stop_time
    last_stop_time = 0
    
    try:
        print("Starting enhanced lane following with grayscale stop...")
        print("Press Ctrl+C to stop")
        
        # Set initial car direction
        px.set_dir_servo_angle(0)
        
        # Wait for camera to initialize
        print("Initializing camera...")
        time.sleep(2)
        
        # Start moving forward
        print("Starting to move forward at speed {}...".format(speed))
        px.forward(speed)
        
        consecutive_no_lanes = 0
        
        while True:
            # First check grayscale sensors for white line crossing
            current_time = time.time()
            if current_time - last_stop_time > MIN_TIME_BETWEEN_STOPS:
                if check_solid_white_line():
                    # Solid white line detected - stop the car
                    print("STOPPING for solid white line crossing...")
                    px.stop()
                    time.sleep(STOP_DURATION)
                    print("Resuming after stop...")
                    last_stop_time = time.time()
                    px.forward(speed)
            
            # Capture frame for lane following
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame for lane detection
            left_lane, right_lane, _ = frameProcess(frame, prev_left_lane, prev_right_lane)
            
            # Update previous lane detections
            if left_lane is not None:
                prev_left_lane = left_lane
            if right_lane is not None:
                prev_right_lane = right_lane
            
            # Lane following logic
            if left_lane is not None or right_lane is not None:
                consecutive_no_lanes = 0
                
                # Calculate steering angle
                steering = calculate_steering(left_lane, right_lane, frame.shape[1])
                
                # Add derivative control
                error = steering
                derivative = error - last_error
                control = kp * error + kd * derivative
                last_error = error
                
                # Limit steering angle
                steering = min(max(control, -max_angle), max_angle)
                
                # Update steering history for smoothing
                steering_history.pop(0)
                steering_history.append(steering)
                
                # Calculate smoothed steering
                smoothed_steering = sum(steering_history) / len(steering_history)
                
                # Calculate speed based on steering angle
                abs_angle = abs(smoothed_steering)
                if abs_angle < 5:
                    current_speed = speed
                elif abs_angle < 10:
                    current_speed = speed * 0.8
                elif abs_angle < 15:
                    current_speed = speed * 0.6
                else:
                    current_speed = speed * 0.4
                
                # Apply steering
                px.set_dir_servo_angle(smoothed_steering)
                
                # Apply appropriate speed
                px.forward(current_speed)
                
                # Print lane info
                if left_lane and right_lane:
                    print("Both lanes: steering={:.2f}, speed={:.1f}".format(
                        smoothed_steering, current_speed))
                elif left_lane:
                    print("Left lane: steering={:.2f}, speed={:.1f}".format(
                        smoothed_steering, current_speed))
                elif right_lane:
                    print("Right lane: steering={:.2f}, speed={:.1f}".format(
                        smoothed_steering, current_speed))
                
            else:
                # No lanes detected
                consecutive_no_lanes += 1
                
                if consecutive_no_lanes > 10:
                    # Lost lanes for too long, go slow and straight
                    px.set_dir_servo_angle(0)
                    px.forward(speed * 0.3)
                    print("No lanes for extended period, very slow straight")
                else:
                    # Continue with last known steering at reduced speed
                    last_steering = steering_history[-1] if steering_history else 0
                    px.set_dir_servo_angle(last_steering * 0.7)
                    px.forward(speed * 0.5)
                    print("No lanes, using last steering: {:.2f}, reduced speed".format(
                        last_steering))
            
            # Processing delay
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
    except Exception as e:
        print("Unexpected error: {}".format(e))
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        px.stop()
        cap.release()
        print("Car stopped and resources released")

def calibrate_grayscale():
    """Calibrate the grayscale sensors"""
    try:
        print("Starting grayscale sensor calibration...")
        print("Move the car over different surfaces to see sensor values")
        print("Press Ctrl+C to exit calibration")
        
        while True:
            # Read grayscale sensor values
            adc_value_list = px.get_grayscale_data()
            
            if adc_value_list:
                left_value = adc_value_list[0]
                middle_value = adc_value_list[1]
                right_value = adc_value_list[2]
                
                print("Sensor values - Left: {}, Middle: {}, Right: {}".format(
                    left_value, middle_value, right_value))
                
                # Show if any sensor would detect white line
                if left_value < GRAYSCALE_THRESHOLD:
                    print("LEFT sensor detects white line!")
                if middle_value < GRAYSCALE_THRESHOLD:
                    print("MIDDLE sensor detects white line!")
                if right_value < GRAYSCALE_THRESHOLD:
                    print("RIGHT sensor detects white line!")
                    
                print("-" * 40)
            else:
                print("Failed to get grayscale sensor data")
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("Calibration stopped by user")
    except Exception as e:
        print("Unexpected error: {}".format(e))
    finally:
        print("Calibration complete")

if __name__ == "__main__":
    try:
        print("\nPicarX Lane Following with Grayscale Stop")
        print("1: Start Lane Following")
        print("2: Calibrate Grayscale Sensors")
        print("3: Quit")
        
        choice = input("Select mode: ")
        
        if choice == "1":
            drive_car()
        elif choice == "2":
            calibrate_grayscale()
        else:
            print("Exiting program")
    except Exception as e:
        print("Fatal error: {}".format(e))
        import traceback
        traceback.print_exc()
        # Make sure car is stopped in case of error
        try:
            px.stop()
        except:
            pass
