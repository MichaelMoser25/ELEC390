import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy.ma.extras import average
import time
from picarx import Picarx  # Import Picarx library

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

    # Removed matplotlib visualization code
    return masked_image

def hough_transform(image):
    # Distance resolution of the accumulator in pixels.
    rho = 1 # 1
    # Angle resolution of the accumulator in radians.
    theta = np.pi / 180
    # Only lines that are greater than threshold will be returned.
    threshold = 20 # 20
    # Line segments shorter than that are rejected.
    minLineLength = 100 # 20
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500 #500
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
    image_processed = regionSelect(image_canny, show=False)  # Always set to False to avoid matplotlib
    lines = hough_transform(image_processed)

    leftLane, rightLane = assignLane(lines, h, prev_left_lane, prev_right_lane)
    
    if leftLane is None or rightLane is None:
        return None, None, image_canny

    # Removed matplotlib visualization code since we're using OpenCV for display

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

def drive_car():
    """Main function to drive the car based on lane detection"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Set a slow, safe speed
    global speed
    speed = 20  # Slow speed for safety
    
    # Initialize previous lane detections
    prev_left_lane = None
    prev_right_lane = None
    
    # Initialize error for derivative control
    last_error = 0
    
    try:
        print("Starting lane following at slow speed...")
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
            left_lane, right_lane, _ = frameProcess(frame, prev_left_lane, prev_right_lane)
            
            # Update previous lane detections
            if left_lane is not None:
                prev_left_lane = left_lane
            if right_lane is not None:
                prev_right_lane = right_lane
                
            # If lanes detected, calculate steering
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
                px.set_dir_servo_angle(steering * 0.8)  # Use 80% of calculated steering for smoother control
                
                # Keep driving at constant slow speed
                px.forward(speed)
                
                print(f"Lane following: error={error}, steering={steering:.2f}")
            else:
                # If no lanes detected, maintain straight direction but reduce speed
                px.set_dir_servo_angle(0)
                px.forward(speed * 0.7)  # Even slower when no lanes detected
                print("No lanes detected, maintaining straight direction")
            
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

def test_lane_detection():
    """Test lane detection without moving the car"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_left_lane = None
    prev_right_lane = None
    
    try:
        print("Starting lane detection test mode...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Process frame for lane detection (without showing matplotlib plots)
            left_lane, right_lane, processed = frameProcess(frame, prev_left_lane, prev_right_lane, show=False)
            
            # Update previous lanes
            if left_lane is not None:
                prev_left_lane = left_lane
            if right_lane is not None:
                prev_right_lane = right_lane
            
            # Draw lanes on display frame if detected
            if left_lane is not None and right_lane is not None:
                try:
                    left_x1, left_x2, y1, y2, _, _ = left_lane
                    right_x1, right_x2, _, _, _, _ = right_lane
                    
                    # Draw left lane line
                    cv2.line(display_frame, (left_x1, y1), (left_x2, y2), (255, 0, 0), 5)
                    # Draw right lane line
                    cv2.line(display_frame, (right_x1, y1), (right_x2, y2), (255, 0, 0), 5)
                    # Draw center line
                    center_x1 = (left_x1 + right_x1) // 2
                    center_x2 = (left_x2 + right_x2) // 2
                    cv2.line(display_frame, (center_x1, y1), (center_x2, y2), (0, 255, 0), 3)
                    
                    # Calculate and display steering
                    steering = calculate_steering(left_lane, right_lane, frame.shape[1])
                    cv2.putText(display_frame, f"Steering: {steering:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Detected lanes - Steering angle: {steering:.2f}")
                except Exception as e:
                    print(f"Error drawing lanes: {e}")
            else:
                cv2.putText(display_frame, "No lanes detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print("No lanes detected")
            
            # Add region of interest overlay
            rows, cols = frame.shape[:2]
            bottom_left = [cols * bottomLeftCol, rows * bottomRow]
            top_left = [cols * topLeftCol, rows * topRow]
            bottom_right = [cols * bottomRightCol, rows * bottomRow]
            top_right = [cols * topRightCol, rows * topRow]
            roi_points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            cv2.polylines(display_frame, roi_points, isClosed=True, color=(0, 255, 255), thickness=2)
            
            # Display the original frame with overlays
            cv2.imshow("Lane Detection", display_frame)
            
            # Display the processed frame (canny edges)
            if processed is not None:
                cv2.imshow("Processed View", processed)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1)  # Process every 100ms for smoother display
    except Exception as e:
        print(f"Error in test mode: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test mode ended")

if __name__ == "__main__":
    try:
        # Set to auto-start line following mode without asking
        print("Auto-starting line following mode at slow speed")
        drive_car()
    except Exception as e:
        print(f"Fatal error: {e}")
        # Make sure car is stopped in case of error
        try:
            px.stop()
        except:
            pass
