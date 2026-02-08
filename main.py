import time
import cv2
import mediapipe as mp
import numpy as np
import winsound

# MediaPipe FaceLandmarker setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_camera():
    """Returns a camera capture object using DirectShow backend"""
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Correct eye landmark indices for 468-point FaceMesh system
# Using the 6-point EAR calculation: [left corner, top left, top right, right corner, bottom right, bottom left]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # Left eye 6 points
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # Right eye 6 points

def detect_gaze_direction(landmarks, w, h):
    """
    Detect gaze direction using eye landmarks and face orientation
    Returns: 'looking_at_screen', 'looking_left', 'looking_right', 'looking_up', 'looking_down', 'unknown'
    """
    try:
        # Get key facial landmarks for gaze estimation
        # Eye center points
        left_eye_center = np.array([
            (landmarks[33].x + landmarks[133].x) * w / 2,
            (landmarks[33].y + landmarks[133].y) * h / 2
        ])
        right_eye_center = np.array([
            (landmarks[362].x + landmarks[263].x) * w / 2,
            (landmarks[362].y + landmarks[263].y) * h / 2
        ])
        
        # Nose tip for head orientation
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
        
        # Eye corners for iris position estimation
        left_eye_outer = np.array([landmarks[33].x * w, landmarks[33].y * h])
        left_eye_inner = np.array([landmarks[133].x * w, landmarks[133].y * h])
        right_eye_inner = np.array([landmarks[362].x * w, landmarks[362].y * h])
        right_eye_outer = np.array([landmarks[263].x * w, landmarks[263].y * h])
        
        # Calculate eye widths
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        
        # Estimate iris positions using pupil landmarks (if available) or eye center
        # Using landmarks 468 (left pupil) and 473 (right pupil) for more accurate gaze
        if len(landmarks) > 473:
            left_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
            right_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
        else:
            # Fallback to eye centers
            left_pupil = left_eye_center
            right_pupil = right_eye_center
        
        # Calculate horizontal gaze using head rotation and eye direction
        # Calculate face center line (nose to forehead)
        forehead_center = np.array([landmarks[9].x * w, landmarks[9].y * h])  # Forehead center
        
        # Vector from forehead to nose
        face_vector = nose_tip - forehead_center
        
        # Calculate head rotation angle
        head_rotation = np.arctan2(face_vector[0], face_vector[1]) * 180 / np.pi
        
        # Method 2: Eye gaze using iris position relative to eye center
        # Calculate how far pupils are from eye centers horizontally
        left_eye_horizontal_offset = left_pupil[0] - left_eye_center[0]
        right_eye_horizontal_offset = right_pupil[0] - right_eye_center[0]
        
        # Normalize by eye width for consistency
        left_gaze_offset = left_eye_horizontal_offset / left_eye_width
        right_gaze_offset = right_eye_horizontal_offset / right_eye_width
        
        # Combine both methods for robust detection
        # Head rotation: -30° = far left, 0° = center, +30° = far right
        # Eye gaze: -0.5 = far left, 0 = center, +0.5 = far right
        
        # Convert head rotation to -1 to 1 scale
        head_gaze_ratio = (head_rotation + 30) / 60  # Normalize -30° to +30° range
        head_gaze_ratio = np.clip(head_gaze_ratio, -1, 1)
        
        # Average eye gaze offsets
        avg_eye_gaze = (left_gaze_offset + right_gaze_offset) / 2
        
        # Combine both methods with weights
        avg_gaze_ratio = (head_gaze_ratio * 0.6) + (avg_eye_gaze * 0.4)
        
        # Convert to 0-1 scale for consistency
        avg_gaze_ratio = (avg_gaze_ratio + 1) / 2
        
        # Calculate vertical gaze using simpler and more reliable method
        # Use the relative position of nose between eyes and chin
        chin_point = np.array([landmarks[152].x * w, landmarks[152].y * h])  # Chin landmark
        
        # Calculate vertical ratios
        eye_to_chin_distance = abs(chin_point[1] - (left_eye_center[1] + right_eye_center[1]) / 2)
        nose_to_eye_distance = abs(nose_tip[1] - (left_eye_center[1] + right_eye_center[1]) / 2)
        
        # Normalize: 0 = nose at eye level, 1 = nose at chin level
        vertical_ratio = nose_to_eye_distance / eye_to_chin_distance if eye_to_chin_distance > 0 else 0
        
        # Debug output for calibration
        print(f"DEBUG GAZE - Horizontal ratio: {avg_gaze_ratio:.3f}, Vertical ratio: {vertical_ratio:.3f}")
        
        # Determine gaze direction with updated sweet spot thresholds
        # Horizontal: 0.65-0.7 = screen, <0.65 = left, >0.7 = right
        # Vertical: 0.125-0.5 = normal range, <0.125 = looking up, >0.5 = looking down
        
        if avg_gaze_ratio < 0.58: # Looking left
            return "looking_left"
        elif avg_gaze_ratio > 0.69: # Looking right
            return "looking_right"
        if vertical_ratio > 0.525:  # Head down
            return "looking_down"
        elif vertical_ratio < 0.16:  # Head up
            return "looking_up"
        # Default to screen for borderline cases
        
        else:
            return "looking_at_screen"
            
    except (IndexError, TypeError, ZeroDivisionError) as e:
        print(f"Error detecting gaze: {e}")
        return "unknown"

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Calculate Eye Aspect Ratio using FaceMesh landmarks"""
    try:
        eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        vertical_2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        # Calculate horizontal distance
        horizontal = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        
        # Prevent division by zero
        if horizontal == 0:
            return 0.0
            
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    except (IndexError, TypeError) as e:
        print(f"Error calculating EAR: {e}")
        return 0.0

def is_user_attentive(gaze_direction, eye_openness_score):
    """
    Check if user is attentive based on gaze direction and eye openness
    Returns: True if attentive, False if distracted
    """
    # User is attentive if looking at screen AND eyes are sufficiently open
    if gaze_direction == "looking_at_screen" and eye_openness_score >= 15:
        return True
    else:
        return False

def analyze_faces():
    """
    Captures video from camera, detects faces using FaceLandmarker, and outputs results to console and video display
    Uses MediaPipe FaceLandmarker with 468 landmarks for detailed facial analysis
    """
    # Create FaceLandmarker options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Get camera
    cap = get_camera()
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Face Analyzer Started - Press 'q' to quit")
    print("-" * 50)
    
    frame_count = 0
    starttime = time.time()

    # Create FaceLandmarker instance
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Calculate timestamp in milliseconds (must be monotonically increasing)
            timestamp_ms = frame_count * 33  # Approximate 30 FPS -> 33 ms per frame
            
            # Detect face landmarks using FaceLandmarker
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Check if faces were detected
            face_detected = False
            num_faces = 0
            
            if result.face_landmarks:
                face_detected = True
                num_faces = len(result.face_landmarks)
                
                h, w, _ = frame.shape
                landmarks = result.face_landmarks[0]

                # Calculate eye aspect ratios
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)

                # Detect gaze direction
                gaze_direction = detect_gaze_direction(landmarks, w, h)

                # Check if user is attentive
                attentive = is_user_attentive(gaze_direction, int((np.clip((left_ear - 0.1) * 400, 0, 100) + np.clip((right_ear - 0.1) * 400, 0, 100)) / 2))

                # Play beep sound if user is not attentive
                if not attentive:
                    try:
                        winsound.Beep(1000, 200)  # 1000Hz frequency, 200ms duration
                    except:
                        pass  # Silently ignore sound errors

                # Debug output to see actual EAR values and gaze
                print(f"DEBUG - Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Gaze: {gaze_direction}, Attentive: {attentive}\n")

                # Normalize EAR to 0–100 per eye (typical EAR range is 0.15-0.35 for open eyes)
                # Eyes closed: ~0.0-0.15, Eyes open: ~0.2-0.35
                eye_openness_score = int((np.clip((left_ear - 0.1) * 400, 0, 100) + np.clip((right_ear - 0.1) * 400, 0, 100)) / 2)

                # Determine gaze status and color
                gaze_status = gaze_direction.replace("_", " ").upper()
                if gaze_direction == "looking_at_screen":
                    gaze_color = (0, 255, 0)  # Green for looking at screen
                    attention_status = "ATTENTIVE"
                else:
                    gaze_color = (0, 165, 255)  # Orange for looking elsewhere
                    attention_status = "DISTRACTED"
                
                cv2.putText(
                    frame,
                    f"Eye Openness: {eye_openness_score}/100",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Gaze: {gaze_status}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    gaze_color,
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Status: {attention_status}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if attention_status == "ATTENTIVE" else (0, 0, 255),
                    2
                )
                
                # Draw attention status box around face
                box_color = (0, 255, 0) if attentive else (0, 0, 255)  # Green if attentive, Red if distracted
                box_thickness = 3
                
                # Draw rectangle around the detected face area
                face_h, face_w = h, w
                cv2.rectangle(frame, (50, 50), (face_w - 50, face_h - 50), box_color, box_thickness)
                
                # Add attention status text on the box
                status_text = "ATTENTIVE" if attentive else "DISTRACTED"
                cv2.putText(frame, status_text, (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                # Draw simple eye landmarks
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Print to console
                print(f"Frame {frame_count}: Face Detected = True | Faces Found: {num_faces} | Eye Openness: {eye_openness_score}/100")
            else:
                print(f"Frame {frame_count}: Face Detected = False")
            
            # Add text overlay on video
            status_text = f"Face Detected: {face_detected}"
            color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display frame count
            cv2.putText(frame, f"Frame: {frame_count}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("Face Detection Analysis", frame)

            frame_count += 1

            # Add delay to reduce FPS to ~10 frames per second
            #time.sleep(0.3)  # 100ms delay = ~3 FPS

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("-" * 100)
    print("Face Analyzer Stopped")
    print(f"Total frames processed: {frame_count}")
    print(f"Ran for approximately {(time.time() - starttime):.2f} seconds")

if __name__ == "__main__":
    analyze_faces()