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


def _extract_z_twist(R_rel: np.ndarray, gain: float = 0.6) -> float:
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
    max_angle = np.deg2rad(75.0)
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


        # ----- 1) 기본값 + 2D 기반 좌우(y) 이동 (기존 코드) -----

        base_x = 0.0   # 앞/뒤 (카메라에서 거리감) – 기본값
        base_z_val = base_z  # 네가 이미 위에서 설정해둔 높이 값 (예: -0.13)

        max_side = 0.25  # 화면 기준 최대 좌우 이동 범위 (미터)

        # 기본 위치 (손 안 보일 때)
        default_y_R = +0.12
        default_y_L = -0.12
        y_R = default_y_R
        y_L = default_y_L

        # 2D 키포인트 기준으로 좌우를 먼저 잡아준다 (fallback)
        if keypoint_2d_R is not None:
            u_R = keypoint_2d_R.landmark[0].x  # 0 ~ 1
            y_R = (u_R - 0.5) * 2.0 * max_side

        if keypoint_2d_L is not None:
            u_L = keypoint_2d_L.landmark[0].x  # 0 ~ 1
            y_L = (u_L - 0.5) * 2.0 * max_side

        # 기본 x는 고정
        x_R = base_x
        x_L = base_x


        # ----- 2) 3D MediaPipe 손목 좌표가 있으면, 그걸로 덮어쓰기 -----

        # 스케일 튜닝용
        scale_xy = 0.4   # 좌우 이동 크기
        scale_x  = 0.4   # 앞뒤 이동 크기

        if wrist_pos_world_R is not None:
            mx_R, my_R, mz_R = wrist_pos_world_R  # Mediapipe world frame (대략 m 단위)

            # mz_R: 카메라에서 멀어질수록 값이 커지는 방향 → SAPIEN x로
            # mx_R: 좌우 방향 → SAPIEN y에 더해주기
            x_R = base_x - mz_R * scale_x
            y_R = default_y_R + mx_R * scale_xy

        if wrist_pos_world_L is not None:
            mx_L, my_L, mz_L = wrist_pos_world_L
            x_L = base_x - mz_L * scale_x
            y_L = default_y_L + mx_L * scale_xy

        # 최종 베이스 위치
        base_pos_right = np.array([x_R, y_R, base_z_val], dtype=float)
        base_pos_left  = np.array([x_L, y_L, base_z_val], dtype=float)



        robot_right.set_qpos(qpos_R[retargeting_to_sapien_R])
        robot_left.set_qpos(qpos_L[retargeting_to_sapien_L])

        robot_right.set_pose(sapien.Pose(base_pos_right, base_quat_right))
        robot_left.set_pose(sapien.Pose(base_pos_left,  base_quat_left))



        # ----------------- RIGHT HAND RETARGETING -----------------
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


            if wrist_R_R is not None and calib_wrist_R_right[0] is not None:
                R_rel_R = wrist_R_R @ calib_wrist_R_right[0].T
                twist_R = _extract_z_twist(R_rel_R, gain=0.6)
                R_base_R = rotations.matrix_from_quaternion(base_quat_right)

                # Z축 기준 회전행렬 만들기
                R_twist_R = rotations.matrix_from_axis_angle(
                    np.array([0.0, 0.0, 1.0, twist_R])
                )

                # 최종 회전 = twist * base
                R_final_R = R_twist_R @ R_base_R
                q_final_R = rotations.quaternion_from_matrix(R_final_R)
            else:
                q_final_R = base_quat_right


            robot_right.set_pose(sapien.Pose(base_pos_right, q_final_R))



        # ----------------- LEFT HAND RETARGETING ------------------
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

            # Fingers
            qpos_L = retargeting_left.retarget(ref_value_L)
            robot_left.set_qpos(qpos_L[retargeting_to_sapien_L])

            # ----- LEFT WRIST -----
            if wrist_R_L is not None and calib_wrist_R_left[0] is not None:
                # 1) 캘리브 기준 상대 회전
                R_rel_L = wrist_R_L @ calib_wrist_R_left[0].T

                # 2) 우리가 정의한 축 제한(예: Z축만, 혹은 단순화된 회전) 적용
                R_rel_L = _simplify_wrist_rotation(R_rel_L)

                # 3) 왼손 기본 자세의 회전행렬 가져오기
                base_T_L = base_pose_left.to_transformation_matrix()
                R_robot0_L = base_T_L[:3, :3]

                # 4) 기본 회전에 상대 회전 덧붙이기
                R_robot_L = R_rel_L @ R_robot0_L

                # 5) 최종 쿼터니언 + 왼손 베이스 위치(base_pos_left)로 pose 설정
                q_robot_L = rotations.quaternion_from_matrix(R_robot_L)
                pose_L = sapien.Pose(base_pos_left, q_robot_L)
                robot_left.set_pose(pose_L)




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
