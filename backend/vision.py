import base64
import numpy as np
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Standard 3D face model points for solvePnP
# Coordinate system: X right, Y down, Z forward
FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (index 1)
    [0.0, 330.0, -65.0],      # Chin (index 152)
    [-225.0, -170.0, -135.0], # Left eye corner (index 33)
    [225.0, -170.0, -135.0],  # Right eye corner (index 263)
    [-150.0, 150.0, -125.0],  # Left Mouth corner (index 61)
    [150.0, 150.0, -125.0]    # Right mouth corner (index 291)
], dtype=np.float64)

def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

def process_frame(frame_data_str: str, client_timestamp: float = None):
    try:
        start_time = time.time()
        if ',' in frame_data_str:
            _, encoded = frame_data_str.split(',', 1)
        else:
            encoded = frame_data_str
            
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        light_intensity = np.mean(gray)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        detection_result = detector.detect(mp_image)
        
        # Base metrics dictionary
        metrics = {
            "timestamp": client_timestamp if client_timestamp else start_time,
            "light_intensity": light_intensity,
            "face_confidence": 0.0,
            "fps": 0.0, # Will be calculated by frontend or aggregator
            "ear": "NaN",
            "eye_openness": "NaN",
            "gaze_pitch": "NaN",
            "gaze_yaw": "NaN",
            "head_pitch": "NaN",
            "head_yaw": "NaN",
            "head_roll": "NaN",
            "eyebrow_tension": "NaN",
            "blink_count": 0, # Aggregator logic
            "blink_duration_ms": 0, # Aggregator logic
        }

        if not detection_result.face_landmarks:
            return {"status": "no_face", "metrics": metrics}

        metrics["face_confidence"] = 1.0
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = img.shape
        
        def get_coords(idx):
            lm = landmarks[idx]
            return [lm.x * w, lm.y * h]

        # EAR Calculation
        left_eye_pts = [get_coords(i) for i in LEFT_EYE]
        right_eye_pts = [get_coords(i) for i in RIGHT_EYE]
        metrics["ear"] = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
        
        # Head Pose Calculation (solvePnP)
        face_2d = np.array([
            get_coords(1),    # Nose
            get_coords(152),  # Chin
            get_coords(33),   # Left Eye
            get_coords(263),  # Right Eye
            get_coords(61),   # Left Mouth
            get_coords(291)   # Right Mouth
        ], dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(
            FACE_3D, face_2d, cam_matrix, dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Gimbal-lock / angle-flip correction
            # RQDecomp3x3 can return roll near ±180° instead of ~0° when
            # solvePnP returns a flipped rotation matrix. Normalise to (-90, 90).
            pitch = float(angles[0])
            yaw   = float(angles[1])
            roll  = float(angles[2])

            # If roll is near ±180°, flip all angles (equivalent pose)
            if abs(roll) > 90:
                pitch = -pitch
                yaw   = -yaw
                roll  = roll - 180 if roll > 0 else roll + 180

            # Clamp to sane display ranges
            metrics["head_pitch"] = float(np.clip(pitch, -90,  90))
            metrics["head_yaw"]   = float(np.clip(yaw,   -90,  90))
            metrics["head_roll"]  = float(np.clip(roll,  -90,  90))

        # Blendshapes extraction (Gaze, Eyebrow, Eye Openness)
        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            bs_dict = {cat.category_name: cat.score for cat in blendshapes}
            
            metrics["eye_openness"] = 1.0 - ((bs_dict.get('eyeBlinkLeft', 0) + bs_dict.get('eyeBlinkRight', 0)) / 2.0)
            metrics["gaze_yaw"] = bs_dict.get('eyeLookOutLeft', 0) - bs_dict.get('eyeLookInLeft', 0) # Approximated
            metrics["gaze_pitch"] = bs_dict.get('eyeLookUpLeft', 0) - bs_dict.get('eyeLookDownLeft', 0)
            metrics["eyebrow_tension"] = (bs_dict.get('browDownLeft', 0) + bs_dict.get('browDownRight', 0) + bs_dict.get('browInnerUp', 0)) / 3.0

        # EAR sanity check
        # EAR < 0.10 with eye_openness > 0.5 = landmark tracking artifact.
        # In this case, use blendshape-derived estimate as authoritative.
        if metrics["ear"] != "NaN" and metrics.get("eye_openness") != "NaN":
            ear_val = float(metrics["ear"])
            bs_open = float(metrics.get("eye_openness", 0.5))
            # Blendshape is more robust — if EAR is implausibly low but eyes are open:
            if ear_val < 0.10 and bs_open > 0.4:
                # Rescale blendshape openness to EAR-equivalent range [0.10, 0.42]
                metrics["ear"] = float(0.10 + bs_open * (0.42 - 0.10))
            # Hard clamp: EAR is physically bounded
            metrics["ear"] = float(np.clip(metrics["ear"], 0.0, 0.6))

        metrics["is_blinking"] = metrics["ear"] < 0.22 if metrics["ear"] != "NaN" else False

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"status": "error", "message": str(e)}
