# Drone_navigation_In_ForestType_env_depthCamera
Drone navigation in a forest-type environment, where the drone detects gaps between tree trunks and passes through them. If an obstacle blocks the path, the drone applies the Velocity Obstacle (VO) method to avoid it and move safely. If no trees are detected for a certain number of frames, the drone initiates landing.
Autonomous Drone Navigation in Forest Environment
This project enables a drone to autonomously navigate through a simulated forest environment, avoiding trees using a combination of computer vision for obstacle detection and advanced flight control. The system is designed for use with the PX4-Autopilot and the Gazebo simulator.

# Overview
The project is composed of three main Python scripts that work in concert:

tree_detector.py: This script uses ROS2 and OpenCV to process RGB and depth images from the drone's camera. It identifies tree trunks based on color (HSV thresholding) and calculates their distance. Its primary function is to detect the widest navigable gap between trees and identify any immediate obstacles.

drone_controller.py: This is the "brain" of the drone. It takes the gap and obstacle information from the tree_detector and translates it into flight commands. It uses the MAVSDK library to communicate with the PX4 flight controller. The controller implements a Velocity Obstacle (VO) method to dynamically adjust the drone's forward and lateral speeds to safely fly through the identified gaps. It also includes fallback behaviors for when the drone gets stuck.

main.py: This is the entry point of the application. It initializes both the tree detector and the drone controller, running them in separate threads to ensure that image processing and flight control can happen concurrently.

# Key Features
Autonomous Navigation: The drone can take off, navigate through a series of obstacles, and land on its own.

Vision-Based Obstacle Avoidance: Uses an RGB-D camera to detect trees and find safe paths.

Dynamic Speed Control: Adjusts its speed based on the width of the gap between trees—slowing down for narrow passages and speeding up for wider ones.

Velocity Obstacle (VO) Method: Implements a safety filter to avoid collisions by adjusting its path when an obstacle is detected in its trajectory.

Stuck Detection & Fallback: Uses optical flow to determine if it's stuck. If so, it initiates a fallback maneuver (sidestep and rotate) to find a new path.

ROS2 and MAVSDK Integration: Seamlessly integrates with standard drone simulation and control tools.

 Setup and Installation
Prerequisites
Ubuntu 22.04 

ROS2 HUMBLE

PX4-Autopilot

Gazebo Simulator with the PX4 SITL (Software In The Loop) environment.

Python 3.10.12

Dependencies
Install the required Python packages using pip:

& pip install numpy opencv-python mavsdk rclpy

Configuration
Camera Intrinsics: The CAMERA_MATRIX in tree_detector.py is configured for the simulated camera. If you use a different camera, you will need to update these values.

HSV Color Range: The lower_trunk_hsv and upper_trunk_hsv values in tree_detector.py are set to detect the color of the tree trunks in the Gazebo simulation. You may need to tune these values for different environments.

ROS Topics: The script subscribes to /camera/image_raw and /camera/depth/image_raw. Ensure your Gazebo simulation is publishing to these topics.

How to Run
Start the PX4-Gazebo Simulation:
Open a terminal and launch the PX4 SITL environment with a world that contains obstacles (e.g., a forest or trees).

# Make sure you are in the PX4-Autopilot directory
& make px4_sitl gazebo

Run the Main Application:
In a new terminal, navigate to the project directory and run the main.py script.

python3 main.py

The drone will then arm, take off, and begin its autonomous navigation routine. An OpenCV window titled "frame" will appear, showing the drone's camera feed with detected obstacles and the current flight state.
