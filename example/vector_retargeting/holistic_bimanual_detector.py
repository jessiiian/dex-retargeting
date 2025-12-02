import mediapipe as mp
import numpy as np
from mediapipe.framework import formats
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Same operator → MANO matrices used before
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class HolisticBimanualDetector:
    """Bimanual detector using MediaPipe Holistic (pose + hands).

    It is designed to be a drop-in replacement for MultiHandDetector:
    returns a list of dicts, each dict has:
        - "handedness": "Right" or "Left" (operator convention, like before)
        - "joint_pos": (21, 3) np.ndarray in MANO-like frame (centered at wrist)
        - "keypoint_2d": NormalizedLandmarkList for drawing
        - "wrist_rot": (3, 3) wrist rotation matrix (more stable using pose+hand)
        - "wrist_pos_world": (3,) np.ndarray, wrist position in world before centering
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        selfie: bool = False,
    ):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie

    # ---------------- Drawing (optional) ----------------

    @staticmethod
    def draw_skeleton_on_image(
        image,
        keypoint_2d_list,
        style: str = "default",
    ):
        if keypoint_2d_list is None:
            return image

        if style == "default":
            drawing_utils = mp.solutions.drawing_utils
            drawing_styles = mp.solutions.drawing_styles
            for keypoint_2d in keypoint_2d_list:
                if keypoint_2d is None:
                    continue
                drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            drawing_utils = mp.solutions.drawing_utils
            for keypoint_2d in keypoint_2d_list:
                if keypoint_2d is None:
                    continue
                drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

        return image

    # ---------------- Public API ----------------

    def detect(self, rgb):
        """Run holistic on an RGB image.

        Args:
            rgb: np.ndarray (H, W, 3), RGB order

        Returns:
            hands: list of dict (like MultiHandDetector), each dict has:
                "handedness", "joint_pos", "keypoint_2d",
                "wrist_rot", "wrist_pos_world"
        """
        results = self.holistic.process(rgb)
        pose_world = results.pose_world_landmarks

        hands = []

        # Right hand
        if results.right_hand_world_landmarks is not None:
            hand_dict_R = self._build_hand_dict(
                hand_world=results.right_hand_world_landmarks,
                hand_2d=results.right_hand_landmarks,
                pose_world=pose_world,
                label="Right",
            )
            if hand_dict_R is not None:
                hands.append(hand_dict_R)

        # Left hand
        if results.left_hand_world_landmarks is not None:
            hand_dict_L = self._build_hand_dict(
                hand_world=results.left_hand_world_landmarks,
                hand_2d=results.left_hand_landmarks,
                pose_world=pose_world,
                label="Left",
            )
            if hand_dict_L is not None:
                hands.append(hand_dict_L)

        return hands

    # ---------------- Internal helpers ----------------

    def _build_hand_dict(
        self,
        hand_world: formats.landmark_pb2.LandmarkList,
        hand_2d: landmark_pb2.NormalizedLandmarkList,
        pose_world: formats.landmark_pb2.LandmarkList,
        label: str,
    ):
        """Build per-hand output dict."""
        if hand_world is None:
            return None

        keypoint_3d_array = self._parse_keypoint_3d(hand_world)  # (21, 3)
        wrist_pos_world = keypoint_3d_array[0].copy()

        # Center at wrist for joint positions
        keypoint_3d_centered = keypoint_3d_array - keypoint_3d_array[0:1, :]

        is_right = (label == "Right")
        wrist_rot = self._estimate_wrist_frame(
            keypoint_3d_array,
            pose_world,
            is_right=is_right,
        )

        # Handedness mapping (same selfie logic as before)
        if self.selfie:
            detected_hand_type = label  # camera is mirrored: keep as is
        else:
            inverse_hand_dict = {"Right": "Left", "Left": "Right"}
            detected_hand_type = inverse_hand_dict[label]

        if detected_hand_type == "Right":
            operator2mano = OPERATOR2MANO_RIGHT
        else:
            operator2mano = OPERATOR2MANO_LEFT

        # joint_pos in MANO-like frame
        joint_pos = keypoint_3d_centered @ wrist_rot @ operator2mano

        return {
            "handedness": detected_hand_type,
            "joint_pos": joint_pos,
            "keypoint_2d": hand_2d,
            "wrist_rot": wrist_rot,
            "wrist_pos_world": wrist_pos_world,
        }

    @staticmethod
    def _parse_keypoint_3d(
        keypoint_3d: formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty((21, 3), dtype=np.float32)
        for i in range(21):
            lm = keypoint_3d.landmark[i]
            keypoint[i, 0] = lm.x
            keypoint[i, 1] = lm.y
            keypoint[i, 2] = lm.z
        return keypoint

    @staticmethod
    def _estimate_wrist_frame(
        hand_key3d: np.ndarray,
        pose_world: formats.landmark_pb2.LandmarkList,
        is_right: bool,
    ) -> np.ndarray:
        """Estimate wrist frame using both pose (elbow–wrist) and hand geometry.

        Returns:
            R: (3, 3) rotation matrix. Columns are [x, y, z] axes.
        """
        assert hand_key3d.shape == (21, 3)
        wrist = hand_key3d[HandLandmark.WRIST.value]

        # If pose is missing, fall back to "hand-only" frame (similar to your old code).
        if pose_world is None:
            points = hand_key3d[[0, 5, 9], :]
            x_vector = points[0] - points[2]
            points_centered = points - np.mean(points, axis=0, keepdims=True)
            _, _, v = np.linalg.svd(points_centered)
            normal = v[2, :]

            x = x_vector - np.sum(x_vector * normal) * normal
            x = x / (np.linalg.norm(x) + 1e-8)
            z = np.cross(x, normal)
            if np.sum(z * (points[1] - points[2])) < 0:
                normal *= -1
                z *= -1
            y = normal / (np.linalg.norm(normal) + 1e-8)
            z = z / (np.linalg.norm(z) + 1e-8)
            R = np.stack([x, y, z], axis=1)
            return R

        # Use elbow → wrist as a stable axis from pose landmarks.
        if is_right:
            wrist_id = PoseLandmark.RIGHT_WRIST.value
            elbow_id = PoseLandmark.RIGHT_ELBOW.value
        else:
            wrist_id = PoseLandmark.LEFT_WRIST.value
            elbow_id = PoseLandmark.LEFT_ELBOW.value

        try:
            pw = pose_world.landmark[wrist_id]
            pe = pose_world.landmark[elbow_id]
            p_wrist_pose = np.array([pw.x, pw.y, pw.z], dtype=np.float32)
            p_elbow_pose = np.array([pe.x, pe.y, pe.z], dtype=np.float32)
            # Forearm direction from elbow to wrist
            forearm = p_wrist_pose - p_elbow_pose
            forearm = forearm / (np.linalg.norm(forearm) + 1e-8)
        except Exception:
            forearm = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Finger direction from wrist to index MCP
        index_mcp = hand_key3d[HandLandmark.INDEX_FINGER_MCP.value]
        finger_dir = index_mcp - wrist
        finger_dir = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)

        # Define:
        #   y-axis ~ forearm direction
        #   z-axis ~ finger direction projected orthogonally to y
        #   x-axis = y × z
        y_axis = forearm
        z_axis = finger_dir - np.dot(finger_dir, y_axis) * y_axis
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return R
