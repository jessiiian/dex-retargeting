import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from pytransform3d import rotations  # For rotation matrix/quaternion conversions

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from multi_hand_detector import MultiHandDetector  # your multi-hand version


# ---------------------------------------------------------------------------
# Wrist rotation utilities
# ---------------------------------------------------------------------------

def _wrist_rot_to_matrix(wrist_rot: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert wrist_rot to a 3x3 rotation matrix.
    - If 3x3: use directly
    - If length 4: treat as quaternion (w, x, y, z)
    - If None or other shape: ignore
    """
    if wrist_rot is None:
        return None
    wrist_rot = np.asarray(wrist_rot)
    if wrist_rot.shape == (3, 3):
        return wrist_rot
    if wrist_rot.shape == (4,):
        return rotations.matrix_from_quaternion(wrist_rot)
    return None


def _extract_z_twist(R_rel: np.ndarray, gain: float = 0.8) -> float:
    """Relative rotation R_rel에서 Z축 방향 twist만 뽑아서 각도로 리턴 (rad).

    - gain으로 크기 조절 (0.0 ~ 1.0 정도)
    - 부호가 반대로 느껴지면 angle_z 앞에 -를 붙이면 됨.
    """
    R_rel = np.asarray(R_rel, dtype=float)
    if R_rel.shape != (3, 3) or not np.all(np.isfinite(R_rel)):
        return 0.0

    # 회전행렬 정규화 (수치 노이즈 날리는 용도)
    U, _, Vt = np.linalg.svd(R_rel)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    # Z축 회전 성분만 추출
    angle_z = np.arctan2(R[1, 0], R[0, 0])  # -pi ~ pi

    # 크기 조절 & 클램프
    angle_z *= gain
    max_angle = np.deg2rad(135.0)
    angle_z = float(np.clip(angle_z, -max_angle, max_angle))
    return angle_z


def _simplify_wrist_rotation(R_rel: np.ndarray) -> np.ndarray:
    """Use only Z-axis rotation from the relative wrist rotation.

    - 안정적이고 축 안 흐트러지게
    - 회전 방향은 '인쪽으로 비틀면 로봇도 인쪽' 이 되도록 부호 정리
    """
    R_rel = np.asarray(R_rel, dtype=float)

    # Sanity check
    if R_rel.shape != (3, 3) or not np.all(np.isfinite(R_rel)):
        return np.eye(3)

    # 살짝 정규화해서 진짜 회전행렬에 가깝게 만들어줌 (숫자 노이즈 방지)
    U, _, Vt = np.linalg.svd(R_rel)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # Z축 회전 성분만 추출
    # 이 각도의 부호를 바꿔서 '인쪽으로 비틀면 로봇도 인쪽'이 되게 조정
    angle_z = -np.arctan2(R[1, 0], R[0, 0])

    # 아주 작은 노이즈는 무시
    if abs(angle_z) < 1e-3:
        return np.eye(3)

    cz = np.cos(angle_z)
    sz = np.sin(angle_z)

    # 순수 Z축 회전 행렬
    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return Rz



# No extra view correction: use robot's initial orientation as-is
WRIST_VIEW_ROT = np.eye(3)


# ---------------------------------------------------------------------------
# Robot loading helper
# ---------------------------------------------------------------------------

def _load_robot_for_config(
    scene: sapien.Scene,
    urdf_path: str,
    xy_offset: np.ndarray,
):
    """Load a robot from URDF, apply scaling and position offset.

    Returns:
        robot: the loaded SAPIEN robot
        base_pose: its initial base pose (used as reference for wrist rotation)
    """
    loader = scene.create_urdf_loader()
    filepath = Path(urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    # Scaling rules depending on robot type
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    # Load GLB-based URDF if exists
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # Adjust initial robot pose to avoid clipping
    if "ability" in robot_name:
        base_z = -0.15
    elif "shadow" in robot_name:
        base_z = -0.2
    elif "dclaw" in robot_name:
        base_z = -0.15
    elif "allegro" in robot_name:
        base_z = -0.05
    elif "bhand" in robot_name:
        base_z = -0.2
    elif "leap" in robot_name:
        base_z = -0.15
    elif "svh" in robot_name:
        base_z = -0.13
    else:
        base_z = -0.15

    base_pos = np.array([0.0, 0.0, base_z]) + xy_offset
    base_pose = sapien.Pose(base_pos)
    robot.set_pose(base_pose)

    return robot, base_pose


# ---------------------------------------------------------------------------
# Main retargeting process (bimanual, 5s auto-calibration)
# ---------------------------------------------------------------------------

def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
):
    """
    Bimanual retargeting (right + left hand) with wrist orientation.
    Wrist calibration:
      - No key press needed.
      - After the viewer starts, we wait 5 seconds.
      - Around 5 seconds, current wrist orientations (for each visible hand)
        are stored as calibration poses.
      - From then on, all wrist rotations are relative to that pose.
    """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(
        f"Start bimanual retargeting with robot={robot_name}, type={retargeting_type}"
    )

    # Load configs for both hands
    cfg_right = RetargetingConfig.load_from_file(
        get_default_config_path(robot_name, retargeting_type, HandType.right)
    )
    cfg_left = RetargetingConfig.load_from_file(
        get_default_config_path(robot_name, retargeting_type, HandType.left)
    )

    retargeting_right = cfg_right.build()
    retargeting_left = cfg_left.build()

    calib_hand_dist = [None]
    base_gap = 0.24  # robot hands default distance (tune as you like)
    base_z = -0.13   # your current ground offset for hands

    gap_state = [base_gap]

    # Single detector that returns both hands
    detector = MultiHandDetector(selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # Scene setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    # scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera (you said you didn't change camera settings, so keep as-is)
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # 1) 처음에는 가운데에 두 손을 거의 같은 위치에 로드
    #    (y 오프셋은 일단 0으로 두고, 나중에 frame 루프에서 우리가 직접 y를 움직일 거야)
    robot_right, base_pose_right = _load_robot_for_config(
        scene, cfg_right.urdf_path, xy_offset=np.array([0.0, 0.0, 0.0])
    )
    robot_left, base_pose_left = _load_robot_for_config(
        scene, cfg_left.urdf_path, xy_offset=np.array([0.0, 0.0, 0.0])
    )

    # 2) 기준 회전(quaternion)은 그대로 써야 하니까 따로 저장
    base_quat_right = base_pose_right.q.copy()
    base_quat_left = base_pose_left.q.copy()

    # 3) 기준 높이(z)도 저장해 두고, 초기 y 오프셋만 살짝 줘서 안 겹치게 해놓기
    base_height = float(base_pose_right.p[2])   # 예: -0.13 같은 값일 것

    default_y_right = +0.12
    default_y_left  = -0.12
    base_x = 0.0

    base_pos_right = np.array([base_x, default_y_right, base_height], dtype=float)
    base_pos_left  = np.array([base_x, default_y_left,  base_height], dtype=float)

    # 4) 시작할 때 한 번 초기 위치/자세 세팅
    robot_right.set_pose(sapien.Pose(base_pos_right, base_quat_right))
    robot_left.set_pose(sapien.Pose(base_pos_left,  base_quat_left))


    # Wrist calibration per hand (None until calibrated)
    calib_wrist_R_right = [None]
    calib_wrist_R_left = [None]

    calib_wrist_pos_right = [None]  # 3D wrist position at calibration (Right)
    calib_wrist_pos_left = [None]   # 3D wrist position at calibration (Left)

    # When did we start? (for 5s auto-calibration)
    start_time = time.time()
    calibration_delay = 5.0  # seconds

    # Mapping from retargeting joint order to SAPIEN joint order (per robot)
    # Right hand
    sapien_joint_names_R = [joint.get_name() for joint in robot_right.get_active_joints()]
    retargeting_joint_names_R = retargeting_right.joint_names
    retargeting_to_sapien_R = np.array(
        [retargeting_joint_names_R.index(name) for name in sapien_joint_names_R]
    ).astype(int)

    # Left hand
    sapien_joint_names_L = [joint.get_name() for joint in robot_left.get_active_joints()]
    retargeting_joint_names_L = retargeting_left.joint_names
    retargeting_to_sapien_L = np.array(
        [retargeting_joint_names_L.index(name) for name in sapien_joint_names_L]
    ).astype(int)

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error("Failed to fetch image from camera in 5 seconds.")
            return

        # Detect both hands
        hands = detector.detect(rgb)

        # Draw all detected hands
        keypoint_2d_list = [h["keypoint_2d"] for h in hands]
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d_list, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)

        # Extract per-hand data
        joint_pos_R = None
        joint_pos_L = None
        wrist_rot_R_raw = None
        wrist_rot_L_raw = None
        keypoint_2d_R = None
        keypoint_2d_L = None
        wrist_pos_world_R = None
        wrist_pos_world_L = None

        for h in hands:
            handedness = h["handedness"]  # "Right" or "Left"
            if handedness == "Right":
                joint_pos_R = h["joint_pos"]
                wrist_rot_R_raw = h["wrist_rot"]
                keypoint_2d_R = h["keypoint_2d"]
                wrist_pos_world_R = h.get("wrist_pos_world", None)
            elif handedness == "Left":
                joint_pos_L = h["joint_pos"]
                wrist_rot_L_raw = h["wrist_rot"]
                keypoint_2d_L = h["keypoint_2d"]
                wrist_pos_world_L = h.get("wrist_pos_world", None)

        wrist_R_R = _wrist_rot_to_matrix(wrist_rot_R_raw)
        wrist_R_L = _wrist_rot_to_matrix(wrist_rot_L_raw)

        # 5s auto-calibration: after delay, use current wrist pose as reference
        elapsed = time.time() - start_time
        if elapsed >= calibration_delay:
            # Right hand
            if calib_wrist_R_right[0] is None and wrist_R_R is not None and joint_pos_R is not None:
                calib_wrist_R_right[0] = wrist_R_R.copy()
                calib_wrist_pos_right[0] = joint_pos_R[0].copy()  # index 0 = wrist
                logger.info("Right wrist orientation calibrated (auto after 5s).")

            # Left hand
            if calib_wrist_R_left[0] is None and wrist_R_L is not None and joint_pos_L is not None:
                calib_wrist_R_left[0] = wrist_R_L.copy()
                calib_wrist_pos_left[0] = joint_pos_L[0].copy()
                logger.info("Left wrist orientation calibrated (auto after 5s).")

        # Keyboard controls: only 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


        # ----- 손 베이스 위치 결정 (양손 사이 거리 과장 버전) -----

        base_x = 0.0                 # 앞/뒤(카메라 방향)
        base_z_val = base_z          # 위에서 정해둔 높이 (예: -0.13)

        # 손 사이 기본 간격 + 얼마나 과장할지
        min_gap  = 0.22              # 손이 거의 붙었을 때 최소 간격 (m)
        gain_gap = 1.8               # 실제 거리(dx)에 곱해서 과장하는 비율
        max_gap  = 0.60              # 너무 멀어지지 않도록 상한 (m)

        # 앞뒤 스케일 (손을 앞으로/뒤로 빼는 동작)
        scale_x = 1.2                # mz → SAPIEN x 로 매핑할 때 크기

        # 기본값: 손이 하나만 보이거나 아예 없을 때
        y_R = +min_gap / 2.0
        y_L = -min_gap / 2.0
        x_R = base_x
        x_L = base_x

        # ----- 1) 두 손목 3D 좌표가 모두 있을 때: "극적으로" 벌어지게 -----
        if wrist_pos_world_R is not None and wrist_pos_world_L is not None:
            mx_R, my_R, mz_R = wrist_pos_world_R
            mx_L, my_L, mz_L = wrist_pos_world_L

            # Mediapipe world 좌표에서 좌우 거리 (x축 차이)
            dx = float(abs(mx_R - mx_L))

            # 손 사이 간격 = 최소 + (실제 거리 * 과장 비율)
            gap = min_gap + gain_gap * dx
            gap = float(np.clip(gap, min_gap, max_gap))

            # 🔍 디버그
            logger.info(f"dx={dx:.4f}, gap={gap:.4f}")

            # 화면 기준 좌우 대칭 배치
            y_R = +gap / 2.0
            y_L = -gap / 2.0

            # 앞/뒤(위/아래 느낌)는 z값으로
            x_R = base_x + mz_R * scale_x
            x_L = base_x + mz_L * scale_x

        else:
            # ----- 2) 한 손만 보이거나, 3D가 없으면: 기존 2D fallback -----
            max_side = 0.25  # 좌우 최대 이동

            default_y_R = +0.12
            default_y_L = -0.12
            y_R = default_y_R
            y_L = default_y_L

            if keypoint_2d_R is not None:
                u_R = keypoint_2d_R.landmark[0].x  # 0 ~ 1
                y_R = (u_R - 0.5) * 2.0 * max_side

            if keypoint_2d_L is not None:
                u_L = keypoint_2d_L.landmark[0].x
                y_L = (u_L - 0.5) * 2.0 * max_side

            # 앞/뒤는 그래도 3D 있으면 써준다
            if wrist_pos_world_R is not None:
                _, _, mz_R = wrist_pos_world_R
                x_R = base_x + mz_R * scale_x
            if wrist_pos_world_L is not None:
                _, _, mz_L = wrist_pos_world_L
                x_L = base_x + mz_L * scale_x

        # 최종 베이스 위치
        base_pos_right = np.array([x_R, y_R, base_z_val], dtype=float)
        base_pos_left  = np.array([x_L, y_L, base_z_val], dtype=float)





        # ----------------- RIGHT HAND RETARGETING -----------------
        q_final_R = base_quat_right  # 기본 회전값

        if joint_pos_R is not None:
            ret_type_R = retargeting_right.optimizer.retargeting_type
            indices_R = retargeting_right.optimizer.target_link_human_indices

            if ret_type_R == "POSITION":
                ref_value_R = joint_pos_R[indices_R, :]
            else:
                origin_indices_R = indices_R[0, :]
                task_indices_R = indices_R[1, :]
                ref_value_R = (
                    joint_pos_R[task_indices_R, :] - joint_pos_R[origin_indices_R, :]
                )

            # Fingers
            qpos_R = retargeting_right.retarget(ref_value_R)
            robot_right.set_qpos(qpos_R[retargeting_to_sapien_R])

            # Wrist rotation (Z축 twist만 사용)
            if wrist_R_R is not None and calib_wrist_R_right[0] is not None:
                R_rel_R = wrist_R_R @ calib_wrist_R_right[0].T
                twist_R = _extract_z_twist(R_rel_R, gain=0.8)

                R_base_R = rotations.matrix_from_quaternion(base_quat_right)
                R_twist_R = rotations.matrix_from_axis_angle(
                    np.array([0.0, 0.0, 1.0, -twist_R])
                )
                R_final_R = R_twist_R @ R_base_R
                q_final_R = rotations.quaternion_from_matrix(R_final_R)

        # 손이 안 잡혔으면 q_final_R은 base_quat_right 그대로

        # 👉 여기서 최종 위치+회전 한 번만 세팅
        robot_right.set_pose(sapien.Pose(base_pos_right, q_final_R))



        # ----------------- LEFT HAND RETARGETING -----------------
        q_final_L = base_quat_left

        if joint_pos_L is not None:
            ret_type_L = retargeting_left.optimizer.retargeting_type
            indices_L = retargeting_left.optimizer.target_link_human_indices

            if ret_type_L == "POSITION":
                ref_value_L = joint_pos_L[indices_L, :]
            else:
                origin_indices_L = indices_L[0, :]
                task_indices_L = indices_L[1, :]
                ref_value_L = (
                    joint_pos_L[task_indices_L, :] - joint_pos_L[origin_indices_L, :]
                )

            qpos_L = retargeting_left.retarget(ref_value_L)
            robot_left.set_qpos(qpos_L[retargeting_to_sapien_L])

            # Wrist rotation (Z축 twist만 사용)
            if wrist_R_L is not None and calib_wrist_R_left[0] is not None:
                R_rel_L = wrist_R_L @ calib_wrist_R_left[0].T
                twist_L = _extract_z_twist(R_rel_L, gain=0.8)

                R_base_L = rotations.matrix_from_quaternion(base_quat_left)
                R_twist_L = rotations.matrix_from_axis_angle(
                    np.array([0.0, 0.0, 1.0, -twist_L])
                )
                R_final_L = R_twist_L @ R_base_L
                q_final_L = rotations.quaternion_from_matrix(R_final_L)

        # 왼손은 회전 고정이면 이거 한 줄로 끝
        robot_left.set_pose(sapien.Pose(base_pos_left, q_final_L))


        # Render a few times for smoother display
        for _ in range(2):
            viewer.render()


# ---------------------------------------------------------------------------
# Frame producer (camera → queue)
# ---------------------------------------------------------------------------
url = "rtsp://"
def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        for _ in range(6):
            cap.grab()

        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,  # kept for CLI compatibility, but ignored (we use both hands)
    camera_path: Optional[str] = None,
):
    """
    Detect the human hand poses from a live camera stream and retarget them
    to two robot hands (right and left) in the same SAPIEN scene.

    Wrist calibration:
        - No key interaction needed.
        - After 5 seconds, the current hand poses (for each visible hand)
          are used as the wrist reference orientation.
    Controls:
        q : quit
    """
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    # maxsize=1 to avoid lag
    queue = multiprocessing.Queue(maxsize=1)

    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(queue, str(robot_dir), robot_name, retargeting_type),
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(1.0)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
