import threading
import cv2
import numpy as np
import rclpy
import logging
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Camera intrinsics for depth camera
CAMERA_MATRIX = np.array([
    [454.68577, 0.0, 424.5],
    [0.0, 454.68577, 240.5],
    [0.0, 0.0, 1.0]
])
FOCAL_LENGTH = CAMERA_MATRIX[0, 0]
IMAGE_CENTER = (424.5, 240.5)
DRONE_WIDTH = 0.47
BASE_SAFETY_MARGIN = 0.2
CLOSE_OBSTACLE_DISTANCE = 1.0
FOV_MARGIN = 50
GAP_CONFIDENCE_FRAMES = 2
MAX_LATERAL_SPEED = 0.9

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class TreeDetector:
    def __init__(self, auto_calibrate_frames=30):
        self.frame = None
        self.depth_frame = None
        self.mutex = threading.Lock()
        self.lower_trunk_hsv = np.array([5, 20, 20])  
        self.upper_trunk_hsv = np.array([20, 255, 200])  
        self.bridge = CvBridge()
        self.prev_gray = None
        self.calibrated = True
        self.auto_calibrate_frames = auto_calibrate_frames
        self.calibration_frames = []
        self.gap_history = []
        self.last_gap_timestamp = 0.0
        self.prev_lateral_velocity = 0.0
        self.alpha = 0.7  # Low-pass filter coefficient

    def image_callback(self, rgb_msg, depth_msg):
        with self.mutex:
            try:
                self.frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
                self.depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
                # Filter out invalid depth values (NaN or infinite)
                self.depth_frame = np.nan_to_num(self.depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                logger.error(f"Image conversion failed: {e}")

    def compute_velocity(self, target_lateral_velocity):
        """Smooth lateral velocity using a low-pass filter."""
        smoothed_lateral = self.alpha * self.prev_lateral_velocity + (1 - self.alpha) * target_lateral_velocity
        self.prev_lateral_velocity = smoothed_lateral
        return smoothed_lateral

    def compute_optical_flow(self):
        with self.mutex:
            if self.frame is None:
                return None, "no_frame"
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return None, "init"
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            self.prev_gray = gray
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_mag = np.mean(mag)
            if avg_mag > 0.5:
                return avg_mag, "moving"
            elif avg_mag < 0.1:
                return avg_mag, "stuck"
            else:
                return avg_mag, "slow"

    def get_gap_info(self, num_levels=3):
        with self.mutex:
            if self.frame is None or self.depth_frame is None or not self.calibrated:
                return None, None, None, None, []

            blurred = cv2.GaussianBlur(self.frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            trunk_mask = cv2.inRange(hsv, self.lower_trunk_hsv, self.upper_trunk_hsv)

            logger.info(f"Non-zero pixels in trunk mask: {np.sum(trunk_mask) / 255}")

            trunk_cnts, _ = cv2.findContours(trunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            trunk_contours = [c for c in trunk_cnts if cv2.contourArea(c) > 2000]

            num_trunks = len(trunk_contours)
            logger.info(f"Number of trunks detected: {num_trunks}")

            close_obstacles = []
            for c in trunk_contours:
                x, y, w, h = cv2.boundingRect(c)
                center_x = x + w / 2 
                center_y = y + h / 2 
                # Ensure center_y and center_x are within bounds
                center_y = min(max(int(center_y), 0), self.depth_frame.shape[0] - 1)
                center_x = min(max(int(center_x), 0), self.depth_frame.shape[1] - 1)
                # Get depth at the center of the contour
                depth = self.depth_frame[center_y, center_x]
                if depth <= 0 or depth > 4.0:
                    continue
                if (depth < CLOSE_OBSTACLE_DISTANCE or
                    center_x < FOV_MARGIN or center_x > (self.frame.shape[1] - FOV_MARGIN)):
                    close_obstacles.append((center_x, depth))

            if num_trunks < 2:
                non_zero_pixels = np.sum(trunk_mask) / 255
                if non_zero_pixels < 100:
                    logger.info("No obstacles detected - safe to land")
                    return "land", trunk_mask, self.frame.copy(), None, close_obstacles
                return None, trunk_mask, self.frame.copy(), None, close_obstacles

            trunk_rects = sorted([cv2.boundingRect(c) for c in trunk_contours], key=lambda r: r[0])
            gaps = []

            for i in range(len(trunk_rects) - 1):
                x1, y1, w1, h1 = trunk_rects[i]
                x2, y2, w2, h2 = trunk_rects[i + 1]
                gap = x2 - (x1 + w1)
                if gap > 0:
                    center_x = x1 + w1 + gap // 2
                    center_y = max(y1 + h1, y2 + h2)  # Use bottom of trunks for depth
                    # Ensure center_y and center_x are within bounds
                    center_y = min(max(int(center_y), 0), self.depth_frame.shape[0] - 1)
                    center_x = min(max(int(center_x), 0), self.depth_frame.shape[1] - 1)
                    depth = self.depth_frame[center_y, center_x]
                    if depth <= 0 or depth > 10.0:  # Ignore invalid depths
                        continue
                    # Calculate gap width in meters
                    pixel_width = gap
                    gap_width_m = (pixel_width * depth) / FOCAL_LENGTH
                    gaps.append((gap_width_m, center_x, depth, (trunk_rects[i], trunk_rects[i + 1])))

            if gaps:
                widest_gap = max(gaps, key=lambda g: g[0])
                logger.info(f"Widest gap detected: width={widest_gap[0]:.2f}m, distance={widest_gap[2]:.2f}m, center_x={widest_gap[1]:.1f}")
                
                current_time = time.time()
                self.gap_history.append((widest_gap, current_time))
                self.gap_history = [(g, t) for g, t in self.gap_history if current_time - t < 1.0]
                if len(self.gap_history) >= GAP_CONFIDENCE_FRAMES:
                    return widest_gap, trunk_mask, self.frame.copy(), widest_gap[3], close_obstacles
                else:
                    logger.info("Waiting for consistent gap detection")
                    return None, trunk_mask, self.frame.copy(), None, close_obstacles
            else:
                return None, trunk_mask, self.frame.copy(), None, close_obstacles

    def draw_and_show(self, state_text="", gap_info=None):
        with self.mutex:
            if self.frame is None or not self.calibrated:
                return
            vis = self.frame.copy()

        hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
        trunk_mask = cv2.inRange(hsv, self.lower_trunk_hsv, self.upper_trunk_hsv)

        contours, _ = cv2.findContours(trunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        for c in contours:
            cv2.drawContours(vis, [c], -1, (0, 0, 255), 2)

        _, _, _, _, close_obstacles = self.get_gap_info()
        if gap_info and isinstance(gap_info, tuple) and len(gap_info) == 4:
            gap_width_m, avg_center_x, avg_distance, _ = gap_info
            pixel_gap = (gap_width_m * FOCAL_LENGTH) / avg_distance  # Convert back to pixels for display
            for y in [160, 200, 240]:
                cv2.line(vis, (int(avg_center_x - pixel_gap // 2), y), (int(avg_center_x + pixel_gap // 2), y), (255, 0, 0), 2)
            cv2.putText(vis, f"Gap={gap_width_m:.2f}m, Dist={avg_distance:.2f}m", (int(avg_center_x - 60), 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif gap_info == "land":
            cv2.putText(vis, "No obstacles detected - Landing", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No Gap Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for center_x, distance in close_obstacles:
            cv2.circle(vis, (int(center_x), int(IMAGE_CENTER[1])), 10, (255, 0, 255), 2)
            cv2.putText(vis, f"Close: {distance:.2f}m", (int(center_x) - 30, int(IMAGE_CENTER[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.putText(vis, f"State: {state_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("frame", vis)
        cv2.waitKey(1)

def ros_image_visualizer(detector):
    try:
        rclpy.init()
        node = rclpy.create_node('tree_detector')
        rgb_sub = Subscriber(node, Image, '/camera/image_raw')
        depth_sub = Subscriber(node, Image, '/camera/depth/image_raw')
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.5)
        ts.registerCallback(detector.image_callback)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
    except Exception as e:
        logger.error(f"ROS image processing failed: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()
