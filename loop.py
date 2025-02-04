

from picarx import Picarx
import time
import random

class Picar:
    def __init__(self):
        self.px = Picarx()
        self.speed = 30  # Increased default speed for better movement
        self.turn_angle = 60  # Default turn angle
        self.move_time = 0.5  # Default movement time

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
        """Helper function for forward movement"""
        self.px.forward(self.speed)
        time.sleep(self.move_time)
        self.stop_and_wait()

    def square_pattern(self):
        """Execute a square movement pattern"""
        try:
            print("Starting square pattern")
            for _ in range(4):  # Complete square
                self.move_forward()
                
                print("Turning right")
                self.turn_right()
            
            return True  # Indicate successful completion
            
        except Exception as e:
            print(f"Error in square pattern: {e}")
            self.stop_and_wait()
            return False

    def adjust_speed(self, increment=5):
        """Adjust movement speed with smaller increments"""
        old_speed = self.speed
        self.speed = min(100, max(20, self.speed + increment))  # Minimum speed of 20
        print(f"Speed adjusted from {old_speed}% to {self.speed}%")
        return self.speed

    def run(self):
        """Main run loop with pattern selection"""
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

                    # Reset direction and pause
                    self.px.set_dir_servo_angle(0)
                    self.stop_and_wait(2)

                    # Occasionally adjust speed
                    if random.random() < 0.3:  # 30% chance
                        self.adjust_speed(random.choice([-5, 5]))

        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        finally:
            self.px.stop()
            print("Motors stopped")

def main():
    try:
        robot = Picar()
        robot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        
if __name__ == "__main__":
    main()

