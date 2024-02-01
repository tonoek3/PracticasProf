
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
import math
import math3d as m3d
import time

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

import URBasic
from functions import move_to_target


# Variable which scales the robot movement from pixels to meters.
m_per_pixel = 00.00009

# Size of the robot view-window
# The robot will at most move this distance in each direction
max_x = 0.2
max_y = 0.2


# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(50)
vert_rot_max = math.radians(25)


def check_max_xy(xy_coord):
    """
    Checks if the face is outside of the predefined maximum values on the lookaraound plane

    Inputs:
        xy_coord: list of 2 values: x and y value of the face in the lookaround plane.
            These values will be evaluated against max_x and max_y

    Return Value:
        x_y: new x and y values
            if the values were within the maximum values (max_x and max_y) these are the same as the input.
            if one or both of the input values were over the maximum, the maximum will be returned instead
    """

    x_y = [0, 0]
    #print("xy before conversion: ", xy_coord)

    if -max_x <= xy_coord[0] <= max_x:
        # checks if the resulting position would be outside of max_x
        x_y[0] = xy_coord[0]
    elif -max_x > xy_coord[0]:
        x_y[0] = -max_x
    elif max_x < xy_coord[0]:
        x_y[0] = max_x
    else:
        raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

    if -max_y <= xy_coord[1] <= max_y:
        # checks if the resulting position would be outside of max_y
        x_y[1] = xy_coord[1]
    elif -max_y > xy_coord[1]:
        x_y[1] = -max_y
    elif max_y < xy_coord[1]:
        x_y[1] = max_y
    else:
        raise Exception(" y is wrong somehow", xy_coord[1], max_y)
    #print("xy after conversion: ", x_y)

    return x_y

def set_lookorigin():
    """
    Creates a new coordinate system at the current robot tcp position.
    This coordinate system is the basis of the face following.
    It describes the midpoint of the plane in which the robot follows faces.

    Return Value:
        orig: math3D Transform Object
            characterises location and rotation of the new coordinate system in reference to the base coordinate system

    """
    position = robot.get_actual_tcp_pose()
    orig = m3d.Transform(position)
    return orig


def move_to_target(list_of_facepos, robot_pos):
    """
    Function that moves the robot to the position of the face

    Inputs:
        list_of_facepos: a list of face positions captured by the camera, only the first face will be used
        robot_pos: position of the robot in 2D - coordinates

    Return Value:
        prev_robot_pos: 2D robot position the robot will move to. The basis for the next call to this funtion as robot_pos
    """


    face_from_center = list(list_of_facepos[0])  # TODO: find way of making the selected face persistent

    prev_robot_pos = robot_pos
    scaled_face_pos = [c * m_per_pixel for c in face_from_center]

    robot_target_xy = [a + b for a, b in zip(prev_robot_pos, scaled_face_pos)]
    # print("..", robot_target_xy)

    robot_target_xy = check_max_xy(robot_target_xy)
    prev_robot_pos = robot_target_xy

    x = robot_target_xy[0]
    y = robot_target_xy[1]
    z = 0
    xyz_coords = m3d.Vector(x, y, z)

    x_pos_perc = x / max_x
    y_pos_perc = y / max_y

    x_rot = x_pos_perc * hor_rot_max
    y_rot = y_pos_perc * vert_rot_max * -1

    tcp_rotation_rpy = [y_rot, x_rot, 0]
    # tcp_rotation_rvec = convert_rpy(tcp_rotation_rpy)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    oriented_xyz = origin * position_vec_coords
    # oriented_xyz_coord = oriented_xyz.get_pose_vector()
    oriented_xyz_coord = oriented_xyz.pose_vector

    coordinates = oriented_xyz_coord

    qnear = robot.get_actual_joint_positions()
    next_pose = coordinates
    robot.set_realtime_pose(next_pose)

    return prev_robot_pos

# model = YOLO("MODELS/best_box.pt")
model = YOLO("MODELS/best_pcb.pt")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

video_resolution = (640, 480)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0] / 2),
                  int(video_resolution[1] / 2))

# Start streaming
pipeline.start(config)

ROBOT_IP = '192.168.0.101'
ACCELERATION = 1  # Robot acceleration value
VELOCITY = 1  # Robot speed value

robot_startposition = (math.radians(0),
                       math.radians(-87),
                       math.radians(-122),
                       math.radians(26),
                       math.radians(93),
                       math.radians(6))

"""PCB TRACKING LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
robot.movej(q=robot_startposition, a=ACCELERATION, v=VELOCITY)

robot_position = [0, 0]
origin = set_lookorigin()

robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1)  # just a short wait to make sure everything is initialised
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            # images = np.hstack((resized_color_image, depth_colormap))
            images = resized_color_image
        else:
            # images = np.hstack((color_image, depth_colormap))
            images = color_image

        result = model.predict(images, conf=0.85)
        found_pcb = len(result)
        pcb_centers = []
        if found_pcb > 0:
            labels = result[0].plot()
            if len(result[0].boxes.xywh) > 0:
                x = float(result[0].boxes.xywh[0][0])
                y = float(result[0].boxes.xywh[0][1])
                w = float(result[0].boxes.xywh[0][2])
                h = float(result[0].boxes.xywh[0][3])
                width = round(w, 2)
                heigth = round(h, 2)

                x_center = int(x)
                y_center = int(y)

                new_distance = depth_frame.get_distance(x_center - 1, y_center - 1) * 1000

                pcb_center = [x_center, y_center]
                position_from_center = (pcb_center[0] - video_midpoint[0], pcb_center[1] - video_midpoint[1])
                pcb_centers.append(position_from_center)

                cv2.line(labels, video_midpoint, pcb_center, (0, 200, 0), 5)
                cv2.circle(labels, pcb_center, 4, (0, 200, 0), 3)

                cv2.putText(labels, f"Z: {round(new_distance, 2)} mm",
                            (pcb_center[0] + 20, pcb_center[1] + 60),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 2), 2)

                cv2.putText(labels, f"W: {width} H:{heigth}",
                            (pcb_center[0] + 20, pcb_center[1] + 40),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 2), 2)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', labels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(pcb_centers) > 0:
                robot_position = move_to_target(pcb_centers, robot_position)
                print("exiting loop")
except KeyboardInterrupt:
    print("closing robot connection")
    # Remember to always close the robot connection, otherwise it is not possible to reconnect
finally:
    # Stop streaming
    pipeline.stop()
    robot.close()
