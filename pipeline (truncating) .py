import cv2
import numpy as np
import pandas as pd
from pupil_apriltags import Detector
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import (
    SensorDataType,
    TimeDomain,
    TimeQueryOptions,
)

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pathlib import Path

import subprocess
import math
import sys
import os


# python <this_file> <path to vrs file>
vrs_file_path = sys.argv[1]
if not Path(vrs_file_path).suffix or Path(vrs_file_path).suffix != ".vrs":
    raise RuntimeError("Incorrect extension")

print(f"Current file path: {vrs_file_path}")

OUTPUT_FPS = 30
OUTPUT_FOLDER = f"output_{vrs_file_path.partition('.')[0]}"

VIDEO_PATHS = {}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # requires python 3.5 btw

print(f"Created folder {OUTPUT_FOLDER}")

vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file_path)
device_calib = vrs_data_provider.get_device_calibration()

# print(f"\nDevice calibration:\n {device_calib}\n")

camera_calib = device_calib.get_camera_calib("camera-rgb")

# get the camera calibration for the rgb camera -> need it for undistortion of the april tag
fx, fy = camera_calib.get_focal_lengths()
cx, cy = camera_calib.get_principal_point()

CAMERA_PARAMS = (fx, fy, cx, cy)
# april tag global data
TAG_SIZE = 0.05
FAMILY = "tag25h9"
OBJECT_TO_DEVICE_COORDS = None

print(f"camera_params = {CAMERA_PARAMS}")


# get hand data
label = "handtracking"
handtracking_stream_id = vrs_data_provider.get_stream_id_from_label(label)
if handtracking_stream_id is None:
    raise RuntimeError("no handtracking stream")


# most of this is straight from the documentation
def create_video_with_hands():
    # helper functions

    def plot_single_hand(
        img, hand_joints_in_device, camera_label, camera_calib, hand_label
    ):
        plot_ratio = 3.0 if camera_label == "camera-rgb" else 1.0

        marker_color = (0, 64, 255) if hand_label == "left" else (0, 255, 255)
        skeleton_color = (0, 255, 0)

        hand_joints_in_camera = []
        for point_in_device in hand_joints_in_device:
            point_in_camera = (
                camera_calib.get_transform_device_camera().inverse() @ point_in_device
            )
            pixel = camera_calib.project(point_in_camera)
            hand_joints_in_camera.append(pixel)

        # draw the lines between joints and circles around

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_joints_in_camera) and end_idx < len(
                hand_joints_in_camera
            ):
                p1 = hand_joints_in_camera[start_idx]
                p2 = hand_joints_in_camera[end_idx]
                if p1 is not None and p2 is not None:
                    cv2.line(
                        img,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        skeleton_color,
                        int(1.5 * plot_ratio),
                    )

        for p in hand_joints_in_camera:
            if p is not None:
                cv2.circle(
                    img, (int(p[0]), int(p[1])), int(3.0 * plot_ratio), marker_color, -1
                )

    # takes in the frame under bgr format, interpolated_hand_pose, camera_label, camera_calib
    def plot_handpose(img, hand_pose, camera_label, camera_calib):
        if hand_pose.left_hand is not None:
            plot_single_hand(
                img,
                hand_pose.left_hand.landmark_positions_device,
                camera_label,
                camera_calib,
                "left",
            )

        if hand_pose.right_hand is not None:
            plot_single_hand(
                img,
                hand_pose.right_hand.landmark_positions_device,
                camera_label,
                camera_calib,
                "right",
            )

    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    rgb_label = "camera-rgb"

    slam_camera_labels = [
        "slam-front-left",
        "slam-front-right",
        "slam-side-left",
        "slam-side-right",
    ]

    rgb_stream_id = vrs_data_provider.get_stream_id_from_label(rgb_label)
    print(f"RGB stream id: {rgb_stream_id}")
    slam_stream_ids = [
        vrs_data_provider.get_stream_id_from_label(label)
        for label in slam_camera_labels
    ]

    deliver_options = vrs_data_provider.get_default_deliver_queued_options()
    deliver_options.deactivate_stream_all()
    for stream_id in slam_stream_ids + [rgb_stream_id]:
        deliver_options.activate_stream(stream_id)

    video_writers = {}
    output_fps = OUTPUT_FPS

    for sensor_data in vrs_data_provider.deliver_queued_sensor_data(deliver_options):
        device_time_ns = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
        image_data_and_record = sensor_data.image_data_and_record()
        stream_id = sensor_data.stream_id()
        camera_label = vrs_data_provider.get_label_from_stream_id(stream_id)
        camera_calib = device_calib.get_camera_calib(camera_label)

        img_array = image_data_and_record[0].to_numpy_array()

        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array.copy()

        if camera_label not in video_writers:
            height, width = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_filename = f"{vrs_file_path[0:5]}_{camera_label}_output.mp4"

            # should move this to the top
            videos_folder_path = os.path.join(OUTPUT_FOLDER, "videos")
            os.makedirs(videos_folder_path, exist_ok=True)
            output_path = os.path.join(videos_folder_path, output_filename)

            VIDEO_PATHS.update({camera_label: output_path})

            video_writers[camera_label] = cv2.VideoWriter(
                output_path, fourcc, output_fps, (width, height)
            )
            print(f"started to write to: {output_path}")

        interpolated_hand_pose = vrs_data_provider.get_interpolated_hand_pose_data(
            handtracking_stream_id, device_time_ns, TimeDomain.DEVICE_TIME
        )

        if interpolated_hand_pose is not None:
            plot_handpose(img_bgr, interpolated_hand_pose, camera_label, camera_calib)

        video_writers[camera_label].write(img_bgr)

    for label, writer in video_writers.items():
        writer.release()


# TO-DO: function to coallate all the footage into on then overlay the RGB on top


# get the mps data - requires uv
def get_mps_data():
    mps_output_folder = os.path.join(OUTPUT_FOLDER, "mps_data")
    os.makedirs(mps_output_folder, exist_ok=True)

    command = [
        "uv",
        "run",
        "gen2_mp_csv_exporter",
        "--vrs-path",
        vrs_file_path,
        "--output-folder",
        mps_output_folder,
    ]

    print(f"Exporting mps data to {mps_output_folder}")

    VIDEO_PATHS.update({"MPS-FOLDER": mps_output_folder})

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Output: {result.stdout}")

    except Exception as e:
        print(f"Error message: {e.__traceback__}")


# VIDEO_PATHS = {
#     "slam-front-right": "output_simple_movement_v1/videos/simpl_slam-front-right_output.mp4",
#     "slam-front-left": "output_simple_movement_v1/videos/simpl_slam-front-left_output.mp4",
#     "slam-side-left": "output_simple_movement_v1/videos/simpl_slam-side-left_output.mp4",
#     "slam-side-right": "output_simple_movement_v1/videos/simpl_slam-side-right_output.mp4",
#     "camera-rgb": "output_simple_movement_v1/videos/simpl_camera-rgb_output.mp4",
# }


# for i in range(provider.get_num_data(rgb_stream_id)):
#     image_data = provider.get_image_data_by_index(rgb_stream_id, i)
#
#     # Get the exact capture timestamp directly from the hardware record
#     current_timestamp_ns = image_data[1].capture_timestamp_ns
#
#     # Convert Aria image to numpy array for cv2
#     frame = image_data[0].to_numpy_array()
#
#     # ... proceed with cv2.cvtColor and detector.detect ...


# def get_april_tag_data():
#     """
#     returns translation matrices device to object
#     will have to use calibration from online_calibration and multiply with open loop slam to get world_object coordinates
#     also ts shi only works for one april tag id atm
#     """

#     # changed it to read the rgb frame so we have access to the same timestamps from the vrs file provider
#     detector = Detector(families=FAMILY)

#     rgb_label = "camera-rgb"
#     rgb_stream_id = vrs_data_provider.get_stream_id_from_label(rgb_label)

#     trajectory_data = []
#     trajectory_data_formatted = []

#     # protection from outlieing values, if the distance between previous and current point is large, we don't add it to the trajectory

#     tx_old, ty_old, tz_old = None, None, None

#     MAX_ALLOWED_JUMP = 1.0

#     for i in range(vrs_data_provider.get_num_data(rgb_stream_id)):
#         image_data = vrs_data_provider.get_image_data_by_index(rgb_stream_id, i)

#         frame = image_data[0].to_numpy_array()
#         current_timestamp_ns = image_data[1].capture_timestamp_ns

#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         results = detector.detect(
#             gray, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE
#         )

#         # if the tag leaves the camera and returns in a far dif position
#         if len(results) == 0:
#             tx_old, ty_old, tz_old = None, None, None
#             continue

#         for r in results:
#             if r.tag_id != 0:
#                 continue
#             tx, ty, tz = r.pose_t.flatten()

#             if tx_old is not None:
#                 # mean square, could extrapolate this to a function that could compute different
#                 distance = math.sqrt(
#                     (tx - tx_old) ** 2 + (ty - ty_old) ** 2 + (tz - tz_old) ** 2
#                 )

#             # I HAD THIS COMMMENTED OUT BEFORE
#                 if distance > MAX_ALLOWED_JUMP:
#                     # changed cause smth was erroring
#                     # tx_old, ty_old, tz_old = tx, ty, tz
#                     print(f"Skipping timestamp {current_timestamp_ns}, jump found")
#                     continue

#             # could calculate roll, pitch and yaw but I don't think we need them

#             num_rgb = vrs_data_provider.get_num_data(rgb_stream_id)
#             num_tags = len(trajectory_data)

#             print(f"RGB frames: {num_rgb}")
#             print(f"AprilTag detections: {num_tags}")
#             print(f"Detection rate: {100*num_tags/num_rgb:.1f}%")

#             transformation_matrix = np.eye(4)
#             transformation_matrix[:3, :3] = r.pose_R
#             transformation_matrix[:3, 3] = r.pose_t.flatten()
#             matrix_str = ",".join(map(str, transformation_matrix.flatten()))

#             trajectory_data.append(
#                 {
#                     "current_timestamp_ns": current_timestamp_ns,
#                     "tag_id": r.tag_id,
#                     "pos_x_meters": tx,
#                     "pos_y_meters": ty,
#                     "pos_z_meters": tz,
#                     "raw_4x4_matrix": matrix_str,
#                 }
#             )

#             for row in transformation_matrix:
#                 trajectory_data_formatted.append(
#                     {
#                         "col1": round(row[0], 2),
#                         "col2": round(row[1], 2),
#                         "col3": round(row[2], 2),
#                         "col4": round(row[3], 2),
#                     }
#                 )

#             tx_old, ty_old, tz_old = tx, ty, tz

#     df = pd.DataFrame(trajectory_data)
#     df_formatted = pd.DataFrame(trajectory_data_formatted)

#     print("Saving april tag CSV data")
#     csv_output_path = os.path.join(OUTPUT_FOLDER, "object_data_device_to_pbject_data")
#     os.makedirs(csv_output_path, exist_ok=True)

#     csv_output_file = os.path.join(
#         csv_output_path, f"{vrs_file_path.partition('.')[0]}_april_tag.csv"
#     )

#     csv_formatted_output_file = os.path.join(
#         csv_output_path, f"{vrs_file_path.partition('.')[0]}_formatted_april_tag.csv"
#     )

#     df_formatted.to_csv(csv_formatted_output_file, index=False)
#     df.to_csv(csv_output_file, index=False)

#     # below shi needs to be worked on will do that later
#     VIDEO_PATHS.update({"APRIL_TAG_ALL": csv_output_file})
#     VIDEO_PATHS.update({"APRIL_TAG_FORMATTED": csv_formatted_output_file})
#     global OBJECT_TO_DEVICE_COORDS
#     OBJECT_TO_DEVICE_COORDS = csv_output_file

def get_april_tag_data():
    """
    returns translation matrices device to object
    will have to use calibration from online_calibration and multiply with open loop slam to get world_object coordinates
    also ts shi only works for one april tag id atm
    """

    # changed it to read the rgb frame so we have access to the same timestamps from the vrs file provider
    detector = Detector(families=FAMILY)

    rgb_label = "camera-rgb"
    rgb_stream_id = vrs_data_provider.get_stream_id_from_label(rgb_label)

    trajectory_data = []
    trajectory_data_formatted = []

    # protection from outlieing values, if the distance between previous and current point is large, we don't add it to the trajectory

    tx_old, ty_old, tz_old = None, None, None

    MAX_ALLOWED_JUMP = 1.0

    for i in range(vrs_data_provider.get_num_data(rgb_stream_id)):
        image_data = vrs_data_provider.get_image_data_by_index(rgb_stream_id, i)

        frame = image_data[0].to_numpy_array()
        current_timestamp_ns = image_data[1].capture_timestamp_ns

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        results = detector.detect(
            gray, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE
        )

        # if the tag leaves the camera and returns in a far dif position
        if len(results) == 0:
            tx_old, ty_old, tz_old = None, None, None
            continue

        for r in results:
            tx, ty, tz = r.pose_t.flatten()

            if tx_old is not None:
                # mean square, could extrapolate this to a function that could compute different
                distance = math.sqrt(
                    (tx - tx_old) ** 2 + (ty - ty_old) ** 2 + (tz - tz_old) ** 2
                )

                if distance > MAX_ALLOWED_JUMP:
                    print(f"Skipping timestamp {current_timestamp_ns}, jump found")
                    continue

            # could calculate roll, pitch and yaw but I don't think we need them

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = r.pose_R
            transformation_matrix[:3, 3] = r.pose_t.flatten()
            matrix_str = ",".join(map(str, transformation_matrix.flatten()))

            trajectory_data.append(
                {
                    "current_timestamp_ns": current_timestamp_ns,
                    "tag_id": r.tag_id,
                    "pos_x_meters": tx,
                    "pos_y_meters": ty,
                    "pos_z_meters": tz,
                    "raw_4x4_matrix": matrix_str,
                }
            )

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = r.pose_R
            transformation_matrix[:3, 3] = r.pose_t.flatten()

            for row in transformation_matrix:
                trajectory_data_formatted.append(
                    {
                        "col1": round(row[0], 2),
                        "col2": round(row[1], 2),
                        "col3": round(row[2], 2),
                        "col4": round(row[3], 2),
                    }
                )

            tx_old, ty_old, tz_old = tx, ty, tz

    df = pd.DataFrame(trajectory_data)
    df_formatted = pd.DataFrame(trajectory_data_formatted)

    print("Saving april tag CSV data")
    csv_output_path = os.path.join(OUTPUT_FOLDER, "object_data_device_to_pbject_data")
    os.makedirs(csv_output_path, exist_ok=True)

    csv_output_file = os.path.join(
        csv_output_path, f"{vrs_file_path.partition('.')[0]}_april_tag.csv"
    )

    csv_formatted_output_file = os.path.join(
        csv_output_path, f"{vrs_file_path.partition('.')[0]}_formatted_april_tag.csv"
    )

    df_formatted.to_csv(csv_formatted_output_file, index=False)
    df.to_csv(csv_output_file, index=False)

    # below shi needs to be worked on will do that later
    VIDEO_PATHS.update({"APRIL_TAG_ALL": csv_output_file})
    VIDEO_PATHS.update({"APRIL_TAG_FORMATTED": csv_output_file})
    global OBJECT_TO_DEVICE_COORDS
    OBJECT_TO_DEVICE_COORDS = csv_output_file

# for open loop and hand tracking
def get_data_rate_for_csvs(file_path):
    # len/timestamp
    pass


# this would be needed only if we want to interpolate in accordance to sth different than obj tracking
def get_reference_frames():
    # load the raw dataset
    # we have to get the frames so we can do the matrix multiplication

    rgb_label = "camera-rgb"

    rgb_stream_id = vrs_data_provider.get_stream_id_from_label(rgb_label)
    rgb_stream_rate = vrs_data_provider.get_nominal_rate_hz(rgb_stream_id)

    # get frame rate of slam
    open_loop_csv_folder_path = os.path.join(VIDEO_PATHS["MPS-FOLDER"], "slam")
    open_loop_cvs_file_path = os.path.join(
        open_loop_csv_folder_path, "open_loop_trajectory.csv"
    )

    april_tag_file_path = OBJECT_TO_DEVICE_COORDS

    april_tag_data = pd.read_csv(april_tag_file_path)
    open_loop_data = pd.read_csv(open_loop_cvs_file_path)

    # get frame rate of hand tracking

    pass


def interpolate_poses(source_times, source_translations, source_quats, target_times):
    interp_trans = interp1d(
        source_times,
        source_translations,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    synced_translations = interp_trans(target_times)

    rotations = R.from_quat(source_quats)
    slerp = Slerp(source_times, rotations)
    synced_rotations = slerp(target_times)

    return synced_translations, synced_rotations.as_quat()


def interpolate_transform():
    # run interpolation to get all of the data at the same frequency so we can multiply the CSVs
    # once we do linear interpolation and see everything working we could try different interpolations to see how that might improve accuracy
    # NOTE: we cannot run simple interpolation on the rotation matrix within the transformation matrix, Sam should know more about this since his background is in math
    # METHODOLOGY: linear interp for translations, spherical interpolation for rotation/quaternions

    april_tag_file = VIDEO_PATHS["APRIL_TAG_ALL"]
    df_tags = pd.read_csv(april_tag_file)
    target_times = df_tags["current_timestamp_ns"].values

    slamy_path = os.path.join(
        VIDEO_PATHS["MPS-FOLDER"], "slam", "open_loop_trajectory.csv"
    )  # or closed loop but that only works with auth from meta on the lab's laptop
    df_slam = pd.read_csv(slamy_path)
    # I saw that there's only microseconds in the slam so we have to transform to nanoseconds to match the object tracking
    df_slam["tracking_timestamp_ns"] = df_slam["tracking_timestamp_us"] * 1000

    hands_path = os.path.join(VIDEO_PATHS["MPS-FOLDER"], "hand_tracking/hand_tracking_results.csv")
    df_hands = pd.read_csv(hands_path)
    df_hands["tracking_timestamp_ns"] = df_hands["tracking_timestamp_us"] * 1000

    # make sure time are within the source, otherwise slerp will crash (got this from the docs)
    valid_mask = (
        (target_times >= df_slam["tracking_timestamp_ns"].min())
        & (target_times <= df_slam["tracking_timestamp_ns"].max())
        & (target_times >= df_hands["tracking_timestamp_ns"].min())
        & (target_times <= df_hands["tracking_timestamp_ns"].max())
    )

    valid_target_times = target_times[valid_mask]

    print(
        f"When interpolating we dropped {len(target_times) - len(valid_target_times)} frames that were out of bounds"
    )

    # scipy excepts quaternions in x, y, z, w format
    slam_times = df_slam["tracking_timestamp_ns"].values
    slam_trans = df_slam[
        ["tx_odometry_device", "ty_odometry_device", "tz_odometry_device"]
    ].values
    slam_quats = df_slam[
        [
            "qx_odometry_device",
            "qy_odometry_device",
            "qz_odometry_device",
            "qw_odometry_device",
        ]
    ].values

    synced_slam_trans, synced_slam_quats = interpolate_poses(
        slam_times, slam_trans, slam_quats, valid_target_times
    )

    df_synced_slam = pd.DataFrame(
        np.hstack(
            (valid_target_times.reshape(-1, 1), synced_slam_trans, synced_slam_quats)
        ),
        columns=[
            "tracking_timestamp_ns",
            "tx_odometry_device",
            "ty_odometry_device",
            "tz_odometry_device",
            "qx_odometry_device",
            "qy_odometry_device",
            "qz_odometry_device",
            "qw_odometry_device",
        ],
    )

    # hands interpolation
    # interpolating just the right hand wrist for now

    hand_times = df_hands["tracking_timestamp_ns"].values
    hand_trans_r = df_hands[
        ["tx_right_device_wrist", "ty_right_device_wrist", "tz_right_device_wrist"]
    ].values
    hand_quats_r = df_hands[
        [
            "qx_right_device_wrist",
            "qy_right_device_wrist",
            "qz_right_device_wrist",
            "qw_right_device_wrist",
        ]
    ].values

    synced_hand_trans_r, synced_hand_quats_r = interpolate_poses(
        hand_times, hand_trans_r, hand_quats_r, valid_target_times
    )

    df_synced_hands_right = pd.DataFrame(
        np.hstack(
            (
                valid_target_times.reshape(-1, 1),
                synced_hand_trans_r,
                synced_hand_quats_r,
            )
        ),
        columns=[
            "tracking_timestamp_ns",
            "tx_right_device_wrist",
            "ty_right_device_wrist",
            "tz_right_device_wrist",
            "qx_right_device_wrist",
            "qy_right_device_wrist",
            "qz_right_device_wrist",
            "qw_right_device_wrist",
        ],
    )

    output_dir = os.path.join(OUTPUT_FOLDER, "synchronized_trajectories")
    os.makedirs(output_dir, exist_ok=True)

    slam_out = os.path.join(output_dir, "synced_slam_data.csv")
    hands_out = os.path.join(output_dir, "synced_hands_data_right.csv")

    df_synced_slam.to_csv(slam_out, index=False)
    df_synced_hands_right.to_csv(hands_out, index=False)

    VIDEO_PATHS.update({"SYNCED_SLAM": slam_out, "SYNCED_HANDS_RIGHT": hands_out})


def get_matrix_from_pose(tx, ty, tz, qx, qy, qz, qw):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[:3, 3] = [tx, ty, tz]
    return mat


def get_pose_from_matrix(mat):
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()
    return t[0], t[1], t[2], q[0], q[1], q[2], q[3]


def compute_reference_frame():
    """
    here we calculate:
    T_hand_tracking = T_device_world (open slam) * T_mps_hand_tracking
    T_world_object = T_device_world (open slam) * T_device_camera (calibration) * T_camera_object (the data from the april tag)
    """
    df_slam = pd.read_csv(VIDEO_PATHS["SYNCED_SLAM"])
    df_hands = pd.read_csv(VIDEO_PATHS["SYNCED_HANDS_RIGHT"])
    df_tags = pd.read_csv(VIDEO_PATHS["APRIL_TAG_ALL"])

    output_dir = os.path.join(OUTPUT_FOLDER, "final_reference")
    os.makedirs(output_dir, exist_ok=True)

    world_slam_data = []
    world_hand_data = []
    world_object_data = []

    formatted_slam = []
    formatted_hand = []
    formatted_object = []

    T_device_camera = camera_calib.get_transform_device_camera().to_matrix()  # for the moment

    for i in range(len(df_slam)):
        ts = df_slam.loc[i, "tracking_timestamp_ns"]

        # slam matrix
        slam_row = df_slam.iloc[i]
        T_device_world = get_matrix_from_pose(
            slam_row["tx_odometry_device"],
            slam_row["ty_odometry_device"],
            slam_row["tz_odometry_device"],
            slam_row["qx_odometry_device"],
            slam_row["qy_odometry_device"],
            slam_row["qz_odometry_device"],
            slam_row["qw_odometry_device"],
        )

        # hand matrix
        hand_row = df_hands.iloc[i]
        T_hand_device = get_matrix_from_pose(
            hand_row["tx_right_device_wrist"],
            hand_row["ty_right_device_wrist"],
            hand_row["tz_right_device_wrist"],
            hand_row["qx_right_device_wrist"],
            hand_row["qy_right_device_wrist"],
            hand_row["qz_right_device_wrist"],
            hand_row["qw_right_device_wrist"],
        )

        tag_matrix_str = df_tags.loc[i, "raw_4x4_matrix"]
        T_camera_object = np.eye(4)
        T_camera_object = np.array(
            [float(x) for x in tag_matrix_str.split(",")]
        ).reshape(4, 4)

        # # try this
        # T_aria_tag = np.array([
        #     [-1,  0,  0,  0], # Flip X (Right -> Left)
        #     [ 0, -1,  0,  0], # Flip Y (Down -> Up)
        #     [ 0,  0,  1,  0], # Keep Z Forward
        #     [ 0,  0,  0,  1]
        # ])
        # corrected_camera_pose = T_aria_tag @ T_camera_object
        # try this

        
        # SLAM is already in world space, so just keep T_device_world
        print(type(T_device_world))
        print(np.shape(T_device_world))
        print("T_world_device")
        print(T_device_world)
        print("T_camera_tag")
        print(T_device_camera)
        T_hand_world = T_device_world @ T_hand_device
        T_object_world = T_device_world @ T_device_camera @ T_camera_object
        print("T_world_object")
        print(T_object_world)

        world_slam_data.append(
            {
                "timestamp_ns": ts,
                **dict(
                    zip(
                        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
                        get_pose_from_matrix(T_device_world),
                    )
                ),
            }
        )
        world_hand_data.append(
            {
                "timestamp_ns": ts,
                **dict(
                    zip(
                        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
                        get_pose_from_matrix(T_hand_world),
                    )
                ),
            }
        )
        world_object_data.append(
            {
                "timestamp_ns": ts,
                **dict(
                    zip(
                        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
                        get_pose_from_matrix(T_object_world),
                    )
                ),
            }
        )

        for row in T_device_world:
            formatted_slam.append(
                {
                    "col1": round(row[0], 2),
                    "col2": round(row[1], 2),
                    "col3": round(row[2], 2),
                    "col4": round(row[3], 2),
                }
            )

        for row in T_hand_world:
            formatted_hand.append(
                {
                    "col1": round(row[0], 2),
                    "col2": round(row[1], 2),
                    "col3": round(row[2], 2),
                    "col4": round(row[3], 2),
                }
            )

        for row in T_object_world:
            formatted_object.append(
                {
                    "col1": round(row[0], 2),
                    "col2": round(row[1], 2),
                    "col3": round(row[2], 2),
                    "col4": round(row[3], 2),
                }
            )

    pd.DataFrame(world_slam_data).to_csv(
        os.path.join(output_dir, "world_slam.csv"), index=False
    )
    pd.DataFrame(world_hand_data).to_csv(
        os.path.join(output_dir, "world_hand.csv"), index=False
    )
    pd.DataFrame(world_object_data).to_csv(
        os.path.join(output_dir, "world_object.csv"), index=False
    )

    pd.DataFrame(formatted_slam).to_csv(
        os.path.join(output_dir, "world_slam_formatted.csv"), index=False
    )
    pd.DataFrame(formatted_hand).to_csv(
        os.path.join(output_dir, "world_hand_formatted.csv"), index=False
    )
    pd.DataFrame(formatted_object).to_csv(
        os.path.join(output_dir, "world_object_formatted.csv"), index=False
    )

    print("Done with the reference stuff ")


def plot_with_mujoco():
    # what mira has been working on
    pass


def plot_with_plotly():
    pass


def compute_screw_trajectory():
    # only after the above functions have been proven to work well
    pass


create_video_with_hands()
get_mps_data()
get_april_tag_data()
# get_reference_frames()
interpolate_transform()
compute_reference_frame()
# print(f"Video paths: {VIDEO_PATHS}")

# find T_world_hands = T_open_loop * T_hand_tracking
# flatten/smooth out the rgb feed
# get april tag data