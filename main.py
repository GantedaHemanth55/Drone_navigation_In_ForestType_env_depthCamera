import threading
import asyncio
import rclpy
from tree_detector import TreeDetector, ros_image_visualizer
from drone_controller import offboard_thread
import cv2

def main():
    detector = TreeDetector()
    vis_thread = threading.Thread(target=ros_image_visualizer, args=(detector,), daemon=True)
    vis_thread.start()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(offboard_thread(detector))
    except KeyboardInterrupt:
        print("[INFO] Manual shutdown requested.")
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
