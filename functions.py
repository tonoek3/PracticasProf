
import numpy as np
import cv2
import imutils
import math3d as m3d


def find_faces_dnn(image):
    """
    Finds human faces in the frame captured by the camera and returns the positions
    uses the pretrained model located at pretrained_model

    Input:
        image: frame captured by the camera

    Return Values:
        face_centers: list of center positions of all detected faces
            list of lists with 2 values (x and y)
        frame: new frame resized with boxes and probabilities drawn around all faces

    """

    frame = image
    frame = imutils.resize(frame, width= video_resolution[0])

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    pretrained_model.setInput(blob)

    # the following line handles the actual face detection
    # it is the most computationally intensive part of the entire program
    # TODO: find a quicker face detection model
    detections = pretrained_model.forward()
    face_centers = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.4:
            continue

        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")


        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
        position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
        face_centers.append(position_from_center)

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #cv2.putText(frame, str(position_from_center), face_center, 0, 1, (0, 200, 0))
        cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
        cv2.circle(frame, face_center, 4, (0, 200, 0), 3)

    return face_centers, frame

def show_frame(frame):
    cv2.imshow('RobotCamera', frame)
    k = cv2.waitKey(6) & 0xff

"""def convert_rpy(angles):

    # This is very stupid:
    # For some reason this doesnt work if exactly  one value = 0
    # the following simply make it a very small value if that happens
    # I do not understand the math behind this well enough to create a better solution
    zeros = 0
    zero_pos = None
    for i,ang in enumerate(angles):
        if ang == 0 :
            zeros += 1
            zero_pos = i
    if zeros == 1:
        #logging.debug("rotation value" + str(zero_pos+1) +"is 0 a small value is added")
        angles[zero_pos] = 1e-6

    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]

    # print ("roll = ", roll)
    # print ("pitch = ", pitch)
    # print ("yaw = ", yaw)
    # print ("")

    for ang in angles:
        # print(ang % np.pi)
        pass

    if roll == pitch == yaw:

        if roll % np.pi == 0:
            rotation_vec = [0, 0, 0]
            return rotation_vec

    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])
    # print("yawmatrix")
    # print(yawMatrix)

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    # print("pitchmatrix")
    # print(pitchMatrix)

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])
    # print("rollmatrix")
    # print(rollMatrix)

    R = yawMatrix * pitchMatrix * rollMatrix
    # print("R")
    # print(R)

    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    # print("theta = ",theta)
    multi = 1 / (2 * math.sin(theta))
    # print("multi = ", multi)


    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    rotation_vec = [rx,ry,rz]
    # print(rx, ry, rz)
    return rotation_vec
"""

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