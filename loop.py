# pip install opencv-python pycoral tflite-runtime


from picarx import Picarx
import time
import random
import numpy as np
import cv2
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import tflite_runtime.interpreter as tflite

class PicarML:
    def __init__(self):
        self.px = Picarx()
        self.speed = 30
        self.turn_angle = 60
        self.move_time = 0.5
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load the TFLite model
        self.model_path = 'traffic_signs_edgetpu.tflite'
        self.interpreter = edgetpu.make_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        # Labels for traffic signs
        self.labels = ['stop_sign', 'one_way']
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        image = cv2.resize(image, (self.width, self.height))
        image = np.expand_dims(image, axis=0)
        image = (image.astype(np.float32) / 127.5) - 1
        return image
        
    def detect_signs(self):
        """Capture image and detect traffic signs"""
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        # Preprocess the image
        processed_image = self.preprocess_image(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        
        # Get detection results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = np.argmax(output_data)
        confidence = output_data[0][prediction]
        
        if confidence > 0.7:  # Confidence threshold
            return self.labels[prediction]
        return None
        
    def handle_sign(self, sign_type):
        """React to detected signs"""
        if sign_type == 'stop_sign':
            print("Stop sign detected! Stopping...")
            self.px.stop()
            time.sleep(3)  # Wait at stop sign
            
        elif sign_type == 'one_way':
            print("One-way sign detected! Adjusting route...")
            self.turn_right()  # Assume we should turn right for one-way
            
    def stop_and_wait(self, wait_time=1):
        """Helper function to stop and wait"""
        self.px.stop()
        time.sleep(wait_time)

    def turn_right(self):
        """Helper function for right turn"""
        self.px.set_dir_servo_angle(self.turn_angle)
        time.sleep(0.5)
        self.px.forward(self.speed)
        time.sleep(3)
        self.stop_and_wait()
        self.px.set_dir_servo_angle(0)
        time.sleep(0.5)

    def move_forward(self):
        """Helper function for forward movement with sign detection"""
        sign = self.detect_signs()
        if sign:
            self.handle_sign(sign)
        else:
            self.px.forward(self.speed)
            time.sleep(self.move_time)
            self.stop_and_wait()

    def square_pattern(self):
        """Execute a square movement pattern with sign detection"""
        try:
            print("Starting square pattern")
            for _ in range(4):
                self.move_forward()
                print("Turning right")
                self.turn_right()
            return True
            
        except Exception as e:
            print(f"Error in square pattern: {e}")
            self.stop_and_wait()
            return False

    def adjust_speed(self, increment=5):
        """Adjust movement speed"""
        old_speed = self.speed
        self.speed = min(100, max(20, self.speed + increment))
        print(f"Speed adjusted from {old_speed}% to {self.speed}%")
        return self.speed

    def run(self):
        """Main run loop with pattern selection and sign detection"""
        patterns = {
            'square': self.square_pattern
        }
        try:
            while True:
                for pattern_name, pattern_func in patterns.items():
                    print(f"\nExecuting {pattern_name} pattern")
                    success = pattern_func()
                    
                    if not success:
                        print("Pattern failed, resetting...")
                        self.px.set_dir_servo_angle(0)
                        self.stop_and_wait(2)
                        continue
                        
                    self.px.set_dir_servo_angle(0)
                    self.stop_and_wait(2)
                    
                    if random.random() < 0.3:
                        self.adjust_speed(random.choice([-5, 5]))
                        
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
        robot = PicarML()
        robot.run()
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
