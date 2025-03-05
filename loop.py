# pip install opencv-python pycoral tflite-runtime requests

from picarx import Picarx
import time
import random
import numpy as np
import cv2
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters import detect
import tflite_runtime.interpreter as tflite
import requests
import threading

class PicarDuckHunter:
    def __init__(self, server_url="http://localhost:8000"):
        # Initialize Picar
        self.px = Picarx()
        self.speed = 30
        self.turn_angle = 60
        self.move_time = 0.5
        
        # Server communication
        self.server_url = server_url
        self.duck_count = 0
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Gripper configuration (adjust based on your hardware)
        self.gripper_open = 0
        self.gripper_closed = 90
        
        # Load the TFLite models
        self.sign_model_path = 'traffic_signs_edgetpu.tflite'
        self.sign_interpreter = edgetpu.make_interpreter(self.sign_model_path)
        self.sign_interpreter.allocate_tensors()
        
        # Duck detection model (COCO SSD or custom model)
        self.duck_model_path = 'coco_ssd_mobilenet_edgetpu.tflite'  # Use COCO SSD or custom duck model
        self.duck_interpreter = edgetpu.make_interpreter(self.duck_model_path)
        self.duck_interpreter.allocate_tensors()
        
        # Get model details
        self.sign_input_details = self.sign_interpreter.get_input_details()
        self.sign_output_details = self.sign_interpreter.get_output_details()
        self.sign_height = self.sign_input_details[0]['shape'][1]
        self.sign_width = self.sign_input_details[0]['shape'][2]
        
        self.duck_input_details = self.duck_interpreter.get_input_details()
        self.duck_output_details = self.duck_interpreter.get_output_details()
        self.duck_height = self.duck_input_details[0]['shape'][1]
        self.duck_width = self.duck_input_details[0]['shape'][2]
        
        # Labels for traffic signs
        self.sign_labels = ['stop_sign', 'one_way']
        
        # For duck detection using COCO SSD
        self.coco_labels = {1: 'person', 2: 'bicycle', 3: 'car', 16: 'bird', 88: 'teddy bear'}
        # Use 'bird' or 'teddy bear' class for ducks, or add a custom class if using custom model
        self.duck_label_id = 16  # bird in COCO
        
        # Line following parameters
        self.line_follow_enabled = True
        self.last_error = 0
        self.kp = 0.5  # Proportional gain
        self.kd = 0.2  # Derivative gain
        
        # Initialize state
        self.is_tracking_duck = False
        
    def preprocess_sign_image(self, image):
        """Preprocess image for sign classification model"""
        image = cv2.resize(image, (self.sign_width, self.sign_height))
        image = np.expand_dims(image, axis=0)
        image = (image.astype(np.float32) / 127.5) - 1
        return image
    
    def preprocess_duck_image(self, image):
        """Preprocess image for duck detection model"""
        image = cv2.resize(image, (self.duck_width, self.duck_height))
        image = np.expand_dims(image, axis=0)
        image = (image.astype(np.float32) / 127.5) - 1
        return image
        
    def detect_signs(self, frame=None):
        """Capture image and detect traffic signs"""
        if frame is None:
            ret, frame = self.camera.read()
            if not ret:
                return None
            
        # Preprocess the image
        processed_image = self.preprocess_sign_image(frame)
        
        # Run inference
        self.sign_interpreter.set_tensor(self.sign_input_details[0]['index'], processed_image)
        self.sign_interpreter.invoke()
        
        # Get detection results
        output_data = self.sign_interpreter.get_tensor(self.sign_output_details[0]['index'])
        prediction = np.argmax(output_data)
        confidence = output_data[0][prediction]
        
        if confidence > 0.7:  # Confidence threshold
            return self.sign_labels[prediction]
        return None
    
    def detect_ducks(self, frame=None):
        """Detect ducks in the image using object detection"""
        if frame is None:
            ret, frame = self.camera.read()
            if not ret:
                return None
        
        # Save original dimensions for bounding box calculation
        orig_height, orig_width = frame.shape[0], frame.shape[1]
        
        # Preprocess the image
        processed_image = self.preprocess_duck_image(frame)
        
        # Run inference
        self.duck_interpreter.set_tensor(self.duck_input_details[0]['index'], processed_image)
        self.duck_interpreter.invoke()
        
        # Get detection results
        # For SSD models, we get boxes, classes, scores, and count
        boxes = self.duck_interpreter.get_tensor(self.duck_output_details[0]['index'])[0]
        classes = self.duck_interpreter.get_tensor(self.duck_output_details[1]['index'])[0]
        scores = self.duck_interpreter.get_tensor(self.duck_output_details[2]['index'])[0]
        count = int(self.duck_interpreter.get_tensor(self.duck_output_details[3]['index'])[0])
        
        duck_boxes = []
        
        for i in range(count):
            if scores[i] > 0.5 and int(classes[i]) == self.duck_label_id:
                # Convert normalized coordinates to pixel values
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * orig_width)
                xmax = int(xmax * orig_width)
                ymin = int(ymin * orig_height)
                ymax = int(ymax * orig_height)
                
                duck_boxes.append((xmin, ymin, xmax, ymax))
        
        return duck_boxes if duck_boxes else None
    
    def detect_line(self, frame=None):
        """Detect line for following"""
        if frame is None:
            ret, frame = self.camera.read()
            if not ret:
                return 0  # No error if can't read frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Define region of interest (bottom part of the image)
        height, width = thresh.shape
        roi = thresh[int(height*0.7):height, :]
        
        # Calculate center of the line
        M = cv2.moments(roi)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            # Calculate error from center
            error = cx - width // 2
            return error
        return 0  # No line detected
        
    def handle_sign(self, sign_type):
        """React to detected signs"""
        if sign_type == 'stop_sign':
            print("Stop sign detected! Stopping...")
            self.px.stop()
            time.sleep(3)  # Wait at stop sign
            
        elif sign_type == 'one_way':
            print("One-way sign detected! Adjusting route...")
            self.turn_right()  # Assume we should turn right for one-way
            
     # ------------------------------------------ if want to implement
    def grab_duck(self):
        """Use gripper to pick up a duck"""
        print("Grabbing duck...")
        
        # Open gripper ------------------------------------------ if want to implement
        self.px.set_pwm_servo_angle(1, self.gripper_open)
        time.sleep(1)
        
        # Move slightly forward to position
        self.px.forward(20)
        time.sleep(0.5)
        self.px.stop()
        
        # Close gripper  ------------------------------------------ if want to implement
        self.px.set_pwm_servo_angle(1, self.gripper_closed)
        time.sleep(1)
        
        # Lift up (if you have a vertical servo)
        self.px.set_pwm_servo_angle(2, 45)
        time.sleep(1)
        
        # Report duck pickup to server
        self.report_duck_pickup()
        
        # Duck count
        self.duck_count += 1
        print(f"Duck captured! Total: {self.duck_count}")
    
    def release_duck(self):
        """Release duck at drop-off point"""
        # Lower gripper (if you have a vertical servo)
        self.px.set_pwm_servo_angle(2, 0)
        time.sleep(1)
        
        # Open gripper
        self.px.set_pwm_servo_angle(1, self.gripper_open)
        time.sleep(1)
    
    def report_duck_pickup(self):
        """Report duck pickup to server"""
        try:
            response = requests.post(f"{self.server_url}/duck-pickup", 
                                     json={"robot_id": "picar-x", "duck_count": self.duck_count})
            if response.status_code == 200:
                print("Duck pickup reported successfully")
            else:
                print(f"Failed to report duck pickup: {response.status_code}")
        except Exception as e:
            print(f"Error reporting duck pickup: {e}")
    
    def track_duck(self, duck_box):
        """Track and move toward a duck"""
        xmin, ymin, xmax, ymax = duck_box
        
        # Calculate center of duck
        duck_center_x = (xmin + xmax) // 2
        
        # Calculate frame center
        frame_center_x = 640 // 2  # Assuming 640x480 resolution
        
        # Calculate error (how far duck is from center)
        error = duck_center_x - frame_center_x
        
        # Duck size (proxy for distance)
        duck_width = xmax - xmin
        duck_height = ymax - ymin
        duck_area = duck_width * duck_height
        
        # Decide how to move based on duck position
        if abs(error) < 50:  # Duck is centered
            if duck_area > 20000:  # Duck is close enough to grab
                print("Duck centered and close - grabbing")
                self.stop_and_wait(0.5)
                self.grab_duck()
                self.is_tracking_duck = False
                return True
            else:  # Duck is centered but far - move forward
                print("Duck centered - moving forward")
                self.px.forward(25)
                return False
        else:  # Duck is not centered - adjust direction
            # Calculate turn angle proportional to error
            turn_angle = min(max(error * 0.1, -self.turn_angle), self.turn_angle)
            print(f"Tracking duck - adjusting direction: {turn_angle}")
            self.px.set_dir_servo_angle(turn_angle)
            self.px.forward(20)
            return False
    
    def follow_line(self):
        """Follow line using PID control"""
        if not self.line_follow_enabled:
            return
            
        ret, frame = self.camera.read()
        if not ret:
            return
            
        # Check for ducks first
        duck_boxes = self.detect_ducks(frame)
        if duck_boxes:
            largest_box = max(duck_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
            self.is_tracking_duck = True
            # IF WE WANT TO DO IT
            if self.track_duck(largest_box):
                # Duck was grabbed, continue with line following
                self.is_tracking_duck = False
                return
            else:
                # Still tracking duck
                return
                
        # Check for signs
        sign = self.detect_signs(frame)
        if sign:
            self.handle_sign(sign)
            return
            
        # If not tracking a duck follow the line
        if not self.is_tracking_duck:
            error = self.detect_line(frame)
            
            # Simple PD controller
            derivative = error - self.last_error
            steering = self.kp * error + self.kd * derivative
            self.last_error = error
            
            # Limit steering angle
            steering = min(max(steering, -self.turn_angle), self.turn_angle)
            
            # Apply steering
            self.px.set_dir_servo_angle(steering)
            self.px.forward(self.speed)
            
    def stop_and_wait(self, wait_time=1):
        """Helper function to stop and wait"""
        self.px.stop()
        time.sleep(wait_time)

    def turn_right(self):
        """Helper function for right turn"""
        self.px.set_dir_servo_angle(self.turn_angle)
        time.sleep(0.5)
        self.px.forward(self.speed)
        time.sleep(1.5)
        self.stop_and_wait()
        self.px.set_dir_servo_angle(0)
        time.sleep(0.5)

    def turn_left(self):
        """Helper function for left turn"""
        self.px.set_dir_servo_angle(-self.turn_angle)
        time.sleep(0.5)
        self.px.forward(self.speed)
        time.sleep(1.5)
        self.stop_and_wait()
        self.px.set_dir_servo_angle(0)
        time.sleep(0.5)

    def adjust_speed(self, increment=5):
        """Adjust movement speed"""
        old_speed = self.speed
        self.speed = min(100, max(20, self.speed + increment))
        print(f"Speed adjusted from {old_speed}% to {self.speed}%")
        return self.speed

    def run(self):
        """Main run loop for line following and duck hunting"""
        try:
            print("Starting duck hunter mode")
            
            # Initialize gripper
            self.px.set_pwm_servo_angle(1, self.gripper_open)
            if hasattr(self.px, 'set_pwm_servo_angle'):
                self.px.set_pwm_servo_angle(2, 0)  # Reset lift servo if we wanna get it going
            
            # Initialize direction
            self.px.set_dir_servo_angle(0)
            
            while True:
                self.follow_line()
                time.sleep(0.05)  # Small delay to prevent CPU overuse
                        
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        finally:
            self.px.stop()
            self.camera.release()
            print("Motors stopped and camera released")

def main():
    try:
        # Change the server URL
        robot = PicarDuckHunter(server_url="http://server-url:port")
        robot.run()
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
