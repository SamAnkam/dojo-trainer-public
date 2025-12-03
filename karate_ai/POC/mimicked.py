import cv2
import numpy as np
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import cdist
import argparse
import os
from dataset_config import DatasetConfig
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Karate Move Similarity Tracker")

    parser.add_argument("--source", type=str, default="webcam",
                        help="Path to video file or 'webcam' for live camera")
    parser.add_argument("--pose_sequence", type=str, default="attention",
                        help="Comma-separated list of karate poses to track (e.g., 'attention,ready,front_kick')")
    parser.add_argument("--show_pose", action="store_true",
                        help="Draw pose skeleton on frames")

    return parser.parse_args()


def load_reference_poses(poses):
    references = defaultdict(list)
    dojo_moves = DatasetConfig()
    for pose in poses:
        ref_files = dojo_moves.get_reference_files_list(move_name=pose) #this will be angle agnostic.

        for ref_file in ref_files:
            data = np.load(ref_file)
            seq = data["landmarks"][..., :3]   # (num_frames, 33, 3)
            seq = seq.reshape(len(seq), -1) #flatten the vector, for faster comparisons

            references[pose].append(seq)

        print(f"Loaded {len(references[pose])} references for pose: {pose}")
    
    return references


def initialize_mediapipe():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    BODY_LANDMARKS = list(range(11,33)) #Shoulders -> Feet
    LANDMARK_INDEX_MAP = {mp_idx: filtered_idx for filtered_idx, mp_idx in enumerate(BODY_LANDMARKS)}

    return pose, BODY_LANDMARKS, LANDMARK_INDEX_MAP, mp_drawing, mp_pose


def canonical_rotate_landmarks(landmarks):
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24

    lm = landmarks.copy()

    left_shoulder = lm[LEFT_SHOULDER][:3]
    right_shoulder = lm[RIGHT_SHOULDER][:3]

    left_hip = lm[LEFT_HIP][:3]
    right_hip = lm[RIGHT_HIP][:3]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    #move hip to origin, for position invariance
    lm[:, :3] -= hip_center
    shoulder_center -= hip_center

    #removing yaw
    up_vec = shoulder_center - hip_center
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)

    #reducing pitch, computing the cross product. note: we're using a right-handed coordinate system
    right_vec = right_hip - left_hip
    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)

    forward_vec = np.cross(up_vec, right_vec)
    forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)

    #recompute right to ensure orthogonality
    right_vec = np.cross(forward_vec, up_vec)
    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)

    #rotation matrix (columns are basis vectors)
    R = np.stack([right_vec, up_vec, forward_vec], axis=1)  # 3×3

    #apply rotation
    rotated = landmarks @ R

    #normalize torso length
    torso = np.linalg.norm(shoulder_center-hip_center)
    if torso > 0:
        rotated /= torso

    return rotated


if __name__ == "__main__":
    args = parse_args()
    poses = args.pose_sequence.split(",") #list of poses, should be in order.

    references = load_reference_poses(poses)

    if args.source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        print("Using webcam stream.")
    else:
        if not os.path.exists(args.source):
            raise FileNotFoundError(f"Video file not found: {args.source}")
        cap = cv2.VideoCapture(args.source)
        print(f"Using video file: {args.source}")
    
    #run loop
    landmark_buffer = [] #sliding window of landmarks, checking partial sequences
    frame_count = 0

    #partial DTW matching
    prefixes = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    similarity_score = 0.0
    window_size = 10 #TODO: experiment with this, aim for 10% of total sequence i.e 30 fps * 10 seconds per sequence = 300/10 = 30 frames

    current_pose_idx = 0
    current_pose_name = poses[current_pose_idx]

    mediapipe_model, BODY_LANDMARKS, LANDMARK_INDEX_MAP, mp_drawing, mp_pose = initialize_mediapipe()
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mediapipe_model.process(frame_rgb)
            
            if results.pose_landmarks:
                #extract pose landmarks
                lm = results.pose_landmarks.landmark
                landmarks = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in BODY_LANDMARKS])

                #error handling; if expected landmarks are not detected, skip frame
                frame_lm = landmarks.shape[0]
                missing = [i for i in (11,12,23,24) if i >= frame_lm]
                if missing:
                    raise ValueError(f"Missing landmarks: {missing}")

                #rectifying the frame
                landmarks = canonical_rotate_landmarks(landmarks)
                landmark_buffer.append(landmarks)

                #keep fixed window length for running dtw comparisons
                if len(landmark_buffer) > window_size:
                    landmark_buffer.pop(0)
                
                if len(landmark_buffer) >= 10 and frame_count % 3 == 0: #every three frames
                    user_action = np.array(landmark_buffer).reshape(len(landmark_buffer), -1)

                    #get all reference sequences for this pose (TODO:we will replace this with a vector search)
                    reference_list = references[current_pose_name]

                    scores = []
                    
                    for reference_sequence in reference_list:

                        reference_length = len(reference_sequence)
                        for pf in prefixes:
                            k = max(5, int(len(reference_sequence) * pf))  # avoid too-short DTW
                            reference_partial = reference_sequence[:k]

                            distance, _ = fastdtw(
                                reference_partial,
                                user_action[-k:],  # align last k user frames
                                dist=lambda x, y: np.linalg.norm(x - y)
                            )

                            sim = np.exp(-distance / k)
                            scores.append(sim)
                        
                    similarity_score = max(scores)

                    
                if similarity_score > 0.5:
                    print(f"pose {current_pose_name} matched with similarity {similarity_score:.2f}")

                    current_pose_idx += 1
                    if current_pose_idx >= len(poses):
                        print("all moves completed")
                        break
                        
                    current_pose_name = poses[current_pose_idx]
                    print(f"next pose: {current_pose_name}")
                
                frame_count += 1

            if args.show_pose:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except ValueError as e:
            print(f"entire body was not detected in the frame. skipping frame.")
        finally:
            # Overlay score
            cv2.putText(frame, f"Pose: {current_pose_name}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.putText(frame, f"Similarity: {similarity_score * 100:.1f}%", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Karate Move Tracking", frame)


cap.release()
cv2.destroyAllWindows()