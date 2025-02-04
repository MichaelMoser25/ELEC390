from picarx import Picarx
import time
import random 

class Picar:
    def __init__(self):
        self.px = Picarx()
        self.speed = 50 # Default speed
        self.current_pattern = 'square'

        def square_pattern(self):
            """Execute a square movement pattern"""

            # Forward
            self.px.forward(self.speed)
            time.sleep(1)

            # Turn right 90 degrees
            self.px.set_dir_servo_angle(90)
            time.sleep(1)

            # Repeat 4 times for a square
            for _ in range(3):
                self.px.forward(self.speed)
                time.sleep(2)
                self.px.set_dir_servo_angle(90)
                time.sleep(1)

    def adjust_speed(self, increment=10):
        """Adjust movement speed"""
        self.speed = min(100, max(0, self.speed + increment))
        return self.speed

    def run(self):
        """Main run loop with pattern selection"""
        patterns = {
            'square': self.square_pattern
        }

        try:
            while True:
                # Cycle through patterns
                for pattern_name, pattern_func in patterns.items():
                    print(f"Executing {pattern_name} pattern... ")
                    pattern_func()

                    # Reset direction and pause between patterns
                    self.px.set_dir_servo_angle(0)
                    self.px.stop()
                    time.sleep(1)

                    # Randomly adjust speed between patterns
                    if random.random() < 0.3:  # 30% chance
                        self.adjust_speed(random.choice([-10, 10]))
                        print(f"Speed adjusted to {self.speed}%")

        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        finally:
            self.px.stop()
            print("Motors stopped")


def main():
    robot = Picar()
    robot.run()


if __name == "__main__":
    main()







