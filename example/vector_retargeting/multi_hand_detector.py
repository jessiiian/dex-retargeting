import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

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


class MultiHandDetector:
    """Detect both hands (up to 2) in one frame and return per-hand data."""

    def __init__(
        self,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.8,
        selfie: bool = False,
    ):
        # Allow up to 2 hands in one frame
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d_list, style: str = "default"
    ):
        """Draw skeletons for all detected hands."""
        if keypoint_2d_list is None:
            return image

        if style == "default":
            for keypoint_2d in keypoint_2d_list:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
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

            for keypoint_2d in keypoint_2d_list:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

        return image

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3D keypoints.
        The output is a 3x3 rotation matrix in the MediaPipe hand frame.
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points_centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points_centered)

        normal = v[2, :]

        # Gram–Schmidt orthonormalization
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # Ensure z direction is consistent with pinky→index direction
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1

        frame = np.stack([x, normal, z], axis=1)
        return frame

    def detect(self, rgb):
        """Detect up to 2 hands in the given RGB image.

        Returns:
            hands: list of dict, each dict has:
                - "handedness": "Right" or "Left" (operator-hand convention)
                - "joint_pos": (21, 3) np.ndarray in MANO-like frame
                - "keypoint_2d": NormalizedLandmarkList for drawing
                - "wrist_rot": (3, 3) rotation matrix in MediaPipe wrist frame
        """
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return []

        hands = []

        # Mediapipe provides: multi_hand_landmarks, multi_hand_world_landmarks, multi_handedness
        for i in range(len(results.multi_hand_landmarks)):
            handedness_proto = results.multi_handedness[i].ListFields()[0][1][0]
            label = handedness_proto.label  # "Right" or "Left" from Mediapipe POV

            # Adjust for selfie / non-selfie convention
            if self.selfie:
                detected_hand_type = label
            else:
                # In non-selfie mode, camera is not mirrored → invert like SingleHandDetector does
                inverse_hand_dict = {"Right": "Left", "Left": "Right"}
                detected_hand_type = inverse_hand_dict[label]

            keypoint_3d = results.multi_hand_world_landmarks[i]
            keypoint_2d = results.multi_hand_landmarks[i]

            keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
            # Make wrist the origin
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

            mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

            # Choose OPERATOR2MANO based on detected hand type
            if detected_hand_type == "Right":
                operator2mano = OPERATOR2MANO_RIGHT
            else:
                operator2mano = OPERATOR2MANO_LEFT

            joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano

            hands.append(
                {
                    "handedness": detected_hand_type,  # "Right" or "Left"
                    "joint_pos": joint_pos,
                    "keypoint_2d": keypoint_2d,
                    "wrist_rot": mediapipe_wrist_rot,
                }
            )

        return hands
