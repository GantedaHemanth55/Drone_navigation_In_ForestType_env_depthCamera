import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from mavsdk.telemetry import LandedState
import math

# Constants and Parameters
DRONE_WIDTH = 0.47  # meters
BASE_SAFETY_MARGIN = 0.2  # meters
MAX_LATERAL_SPEED = 0.9  # m/s
MIN_FORWARD_SPEED = 0.2  # m/s
MAX_FORWARD_SPEED = 1.4  # m/s
DEFAULT_FORWARD_OFFSET = 0.3  # m/s
NORMAL_ALTITUDE = -0.8  # meters (NED)
ALTITUDE_CHANGE_SPEED = 0  # m/s
FALLBACK_LATERAL_SHIFT = 0.5  # meters
FALLBACK_YAW_ANGLE = 30.0 
FOCAL_LENGTH = 454.68577 
IMAGE_CENTER = (424.5, 240.5)  
LANDING_TIMEOUT = 20  # seconds
GAP_CONFIDENCE_FRAMES = 1


def apply_velocity_obstacle_filter(vx, vy, obstacle_list, drone_pos=(0, 0), safety_radius=0.5):
    from math import hypot, atan2, asin
    
    # Check for close obstacles requiring immediate sidestep
    closest_obstacle_info = None
    min_dist_for_sidestep = float('inf')

    for ox, oy in obstacle_list:
        rel_x = ox - drone_pos[0]
        rel_y = oy - drone_pos[1]
        dist = hypot(rel_x, rel_y)
        if dist < min_dist_for_sidestep:
            min_dist_for_sidestep = dist
            closest_obstacle_info = (rel_x, rel_y)
    

    if min_dist_for_sidestep <= 0.3:
        print(f"SIDE STEP: Obstacle at distance {min_dist_for_sidestep:.2f}m. Forcing sidestep.")
        rel_x, _ = closest_obstacle_info
        sidestep_vy = MAX_LATERAL_SPEED if rel_x <= 0 else -MAX_LATERAL_SPEED
        return 0.0, sidestep_vy


    blocked = False
    dynamic_radius = safety_radius + 0.4 * np.hypot(vx, vy)

    closest_obstacle = None
    min_dist = float('inf')
    for ox, oy in obstacle_list:
        rel_x = ox - drone_pos[0]
        rel_y = oy - drone_pos[1]
        dist = hypot(rel_x, rel_y)
        if dist < 0.1: continue
        if dist < min_dist:
            min_dist = dist
            closest_obstacle = (rel_x, rel_y)

        vx_mag = np.hypot(vx, vy)
        if vx_mag < 0.1: continue

        angle_to_obs = atan2(rel_y, rel_x)
        velocity_angle = atan2(vy, vx)
        angle_diff = abs(angle_to_obs - velocity_angle)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        cone_half_angle = asin(min(dynamic_radius / dist, 1.0))

        if angle_diff < cone_half_angle:
            print(f"VO: Obstacle at ({ox:.2f}, {oy:.2f}) blocks path. Distance: {dist:.2f}m")
            blocked = True
            break

    if blocked and closest_obstacle:
        rel_x, _ = closest_obstacle
        print("VO: Trying sidestep...")
        sidestep_vy = MAX_LATERAL_SPEED if rel_x <= 0 else -MAX_LATERAL_SPEED
        return 0.0, sidestep_vy

    return vx, vy

# Rotate velocities based on yaw
def rotate_velocity(vx, vy, yaw_rad):
    """Rotate velocity vector (vx, vy) by yaw angle (radians) to align with drone's heading."""
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    vx_ned = vx * cos_yaw - vy * sin_yaw
    vy_ned = vx * sin_yaw + vy * cos_yaw
    return vx_ned, vy_ned

# Main Drone Control Logic
async def offboard_thread(detector):
    drone = System()
    try:
        await drone.connect(system_address="udp://:14540")
        print(" Waiting for connection...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("Connected to drone")
                break
    except Exception as e:
        print(f"Failed to connect to drone: {e}")
        return

    print("Waiting for global and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Health OK")
            break

    # Capture initial yaw angle before takeoff
    initial_yaw_deg = 0.0
    async for attitude in drone.telemetry.attitude_euler():
        initial_yaw_deg = attitude.yaw_deg
        print(f"Initial yaw captured: {initial_yaw_deg:.2f} degrees")
        break
    initial_yaw_rad = math.radians(initial_yaw_deg)

    print("Arming...")
    try:
        await drone.action.arm()
    except Exception as e:
        print(f"Failed to arm drone: {e}")
        return

    print(f"Taking off to {abs(NORMAL_ALTITUDE)}m altitude...")
    try:
        await drone.action.set_takeoff_altitude(abs(NORMAL_ALTITUDE))
        await drone.action.takeoff()
        await asyncio.sleep(10)
    except Exception as e:
        print(f"Takeoff failed: {e}")
        await drone.action.disarm()
        return

    print("Entering offboard mode...")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, initial_yaw_deg))
        await drone.offboard.start()
        print("Offboard mode started")
    except OffboardError as e:
        print(f"Failed to start offboard mode: {e._result.result}")
        await drone.action.disarm()
        return

    print("Starting vision-based control loop...")
    stuck_counter = 0
    fallback_active = False
    fallback_direction = 1
    no_trunk_counter = 0
    state = "navigate"
    altitude = NORMAL_ALTITUDE

    async for position in drone.telemetry.position_velocity_ned():
        altitude = position.position.down_m
        break

    while True:
        gap_info, mask, frame_copy, gap_rects, close_obstacles = detector.get_gap_info()
        flow_mag, flow_status = detector.compute_optical_flow()
        state_text = f"Flow: {flow_status}"

        if flow_status == "stuck":
            stuck_counter += 1
        else:
            stuck_counter = 0
            if state == "fallback":
                state = "navigate"

        if stuck_counter > 50 and not fallback_active:
            print("Stuck too long! Initiating fallback...")
            fallback_active = True
            state = "fallback"
            lateral_shift = FALLBACK_LATERAL_SHIFT * fallback_direction
            print(f"Shifting sideways: {lateral_shift:.1f} m")
            try:
                vx_ned, vy_ned = rotate_velocity(0.0, lateral_shift, initial_yaw_rad)
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg))
                await asyncio.sleep(2.0)
                print(f"Rotating yaw by {FALLBACK_YAW_ANGLE * fallback_direction}° relative to initial yaw")
                target_yaw = initial_yaw_deg + (fallback_direction * FALLBACK_YAW_ANGLE)
                await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, target_yaw))
                await asyncio.sleep(3.0)
            except OffboardError as e:
                print(f"Fallback failed: {e._result.result}")
            fallback_direction *= -1
            stuck_counter = 0
            fallback_active = False
            print("Rescanning after fallback")

        if gap_info == "land":
            no_trunk_counter += 1
            if no_trunk_counter > GAP_CONFIDENCE_FRAMES:
                state = "landing"
                state_text += " | Landing initiated"
                try:
                    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, initial_yaw_deg))
                    await asyncio.sleep(1)
                    await drone.action.land()
                    break
                except Exception as e:
                    print(f"Landing failed: {e}")
                    break
            else:
                state_text += " | No trunks: slow forward"
                vx_ned, vy_ned = rotate_velocity(DEFAULT_FORWARD_OFFSET, 0.0, initial_yaw_rad)
                velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)
        elif gap_info is None or not isinstance(gap_info, tuple):
            no_trunk_counter += 1
            if close_obstacles:
                state_text += " | Single obstacle: applying VO"
                obstacle_list = []
                very_close = False
                for center_x, distance in close_obstacles:
                    rel_x = ((center_x - IMAGE_CENTER[0]) / FOCAL_LENGTH) * distance
                    rel_y = distance
                    obstacle_list.append((rel_x, rel_y))
                    if distance < 0.5:
                        very_close = True
                if very_close:
                    state_text += " | Very close obstacle: stopping"
                    velocity_cmd = VelocityNedYaw(0.0, 0.0, 0.0, initial_yaw_deg)
                else:
                    forward_speed = DEFAULT_FORWARD_OFFSET
                    boosted_lateral = MAX_LATERAL_SPEED * 0.45
                    safe_vx, safe_vy = apply_velocity_obstacle_filter(forward_speed, boosted_lateral, obstacle_list)
                    vx_ned, vy_ned = rotate_velocity(safe_vx, safe_vy, initial_yaw_rad)
                    if safe_vx == 0.0 and abs(safe_vy) == MAX_LATERAL_SPEED:
                        state_text += " | Sidestepping single obstacle"
                    else:
                        state_text += " | Moving forward safely"
                    velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)
            else:
                if flow_status == "moving":
                    state_text += " | Forward: moving"
                    vx_ned, vy_ned = rotate_velocity(MAX_FORWARD_SPEED, 0.0, initial_yaw_rad)
                    velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)
                elif flow_status == "stuck":
                    state_text += " | Stuck: climbing"
                    velocity_cmd = VelocityNedYaw(0.0, 0.0, ALTITUDE_CHANGE_SPEED, initial_yaw_deg)
                else:
                    state_text += " | Slow forward"
                    vx_ned, vy_ned = rotate_velocity(DEFAULT_FORWARD_OFFSET, 0.0, initial_yaw_rad)
                    velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)
        else:
            no_trunk_counter = 0
            gap_width_m, gap_cx, distance, gap_rects = gap_info

            lateral_offset_m = ((gap_cx - IMAGE_CENTER[0]) / FOCAL_LENGTH) * distance
            safety_margin = BASE_SAFETY_MARGIN + 0.1 * abs(detector.prev_lateral_velocity)
            print(f"Detected gap width: {gap_width_m:.2f} meters, Safety Margin: {safety_margin:.2f} m")

            lateral_velocity = np.clip(lateral_offset_m * 1.5, -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED)
            lateral_velocity = detector.compute_velocity(lateral_velocity)

            min_safe_gap = DRONE_WIDTH + safety_margin
            max_safe_gap = 2.0
            if gap_width_m <= min_safe_gap:
                forward_speed = MIN_FORWARD_SPEED
            elif gap_width_m >= max_safe_gap:
                forward_speed = MAX_FORWARD_SPEED
            else:
                forward_speed = MIN_FORWARD_SPEED + (gap_width_m - min_safe_gap) * (
                    (MAX_FORWARD_SPEED - MIN_FORWARD_SPEED) / (max_safe_gap - min_safe_gap)
                )

            obstacle_list = []
            if gap_rects:
                for tree in gap_rects:
                    x, y, w, h = tree
                    cx = x + w / 2
                    rel_y = distance
                    rel_x = ((cx - IMAGE_CENTER[0]) / FOCAL_LENGTH) * distance
                    obstacle_list.append((rel_x, rel_y))
            for center_x, dist_obs in close_obstacles:
                rel_x = ((center_x - IMAGE_CENTER[0]) / FOCAL_LENGTH) * dist_obs
                obstacle_list.append((rel_x, dist_obs))

            safe_vx, safe_vy = apply_velocity_obstacle_filter(forward_speed, lateral_velocity, obstacle_list)

            if safe_vx == 0.0 and abs(safe_vy) == MAX_LATERAL_SPEED:
                state_text += " | VO sidestep"
                vx_ned, vy_ned = rotate_velocity(safe_vx, safe_vy, initial_yaw_rad)
                velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)
            else:
                state_text += f" | Gap: {gap_width_m:.2f}m"
                vx_ned, vy_ned = rotate_velocity(safe_vx, safe_vy, initial_yaw_rad)
                velocity_cmd = VelocityNedYaw(vx_ned, vy_ned, 0.0, initial_yaw_deg)

        detector.draw_and_show(state_text=state_text, gap_info=gap_info)
        try:
            await drone.offboard.set_velocity_ned(velocity_cmd)
        except OffboardError as e:
            print(f"Velocity command failed: {e._result.result}")

        # Inside your loop or after trunk clearing
        if no_trunk_counter > 100:
            print("Trunks cleared. Landing...")
            try:
                await drone.action.land()
                # Wait for landing to complete by checking telemetry
                print("Waiting for drone to land...")
                async for state in drone.telemetry.landed_state():
                    if state == LandedState.ON_GROUND:
                        print("Drone confirmed landed")
                        break
                    await asyncio.sleep(0.5)  # Check every 0.5 seconds
            except Exception as e:
                print(f"Landing failed: {e}")
                # Handle landing failure (e.g., retry or exit safely)
                return  # Exit to avoid disarming if landing failed
        
