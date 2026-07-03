import cv2
import mediapipe as mp
import math

class GazeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Crucial for Iris tracking (Landmarks 468 & 473)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices defined by MediaPipe
        self.RIGHT_EYE_OUTER = 33
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_IRIS = 468
        
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 263
        self.LEFT_IRIS = 473

    def euclidean_distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

    def get_iris_position(self, iris_center, outer_point, inner_point):
        total_distance = self.euclidean_distance(outer_point, inner_point)
        iris_distance = self.euclidean_distance(outer_point, iris_center)
        ratio = iris_distance / total_distance if total_distance > 0 else 0.5
        
        if ratio <= 0.40:
            return "RIGHT"
        elif ratio >= 0.60:
            return "LEFT"
        else:
            return "CENTER"

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        cheat_flag = False
        message = "Focus: PERFECT"
        color = (0, 255, 0) # Default Green

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # ==========================================
                # ૧. Advanced Iris Tracking
                # ==========================================
                r_iris = landmarks[self.RIGHT_IRIS]
                r_outer = landmarks[self.RIGHT_EYE_OUTER]
                r_inner = landmarks[self.RIGHT_EYE_INNER]
                
                l_iris = landmarks[self.LEFT_IRIS]
                l_inner = landmarks[self.LEFT_EYE_INNER]
                l_outer = landmarks[self.LEFT_EYE_OUTER]
                
                right_gaze = self.get_iris_position(r_iris, r_outer, r_inner)
                left_gaze = self.get_iris_position(l_iris, l_outer, l_inner)
                
                if right_gaze != "CENTER" or left_gaze != "CENTER":
                    cheat_flag = True
                    direction = right_gaze if right_gaze != "CENTER" else left_gaze
                    message = f"ALERT: Eye Looking {direction}!"
                    color = (0, 0, 255) # Red for cheating

                # ==========================================
                # ૨. Head Pose Tracking (માથું ફેરવવાની વોર્નિંગ)
                # ==========================================
                nose_tip = landmarks[1]
                left_edge = landmarks[234]
                right_edge = landmarks[454]

                left_dist = (nose_tip.x - left_edge.x) * w
                right_dist = (right_edge.x - nose_tip.x) * w

                if right_dist > 0: 
                    head_ratio = left_dist / right_dist
                    if head_ratio > 1.6:
                        cheat_flag = True
                        message = "ALERT: Head Turned Right!"
                        color = (0, 0, 255)
                    elif head_ratio < 0.65:
                        cheat_flag = True
                        message = "ALERT: Head Turned Left!"
                        color = (0, 0, 255)

                # ==========================================
                # ૩. આંખની કીકી પર ભૂરા રંગના ટપકાં (Blue Dots)
                # ==========================================
                rx, ry = int(r_iris.x * w), int(r_iris.y * h)
                lx, ly = int(l_iris.x * w), int(l_iris.y * h)
                cv2.circle(frame, (rx, ry), 4, (255, 0, 0), -1) 
                cv2.circle(frame, (lx, ly), 4, (255, 0, 0), -1) 

        else:
            # ચહેરો કેમેરાની બહાર જાય ત્યારે એલર્ટ
            cheat_flag = True
            message = "CRITICAL: Face Missing!"
            color = (0, 0, 255)

        return frame, cheat_flag, message, color

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()
    
    print("Advanced Iris & Head Tracker Active... Press 'ESC' to exit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # મિરર ઇફેક્ટ
        image = cv2.flip(image, 1)

        processed_img, is_cheating, msg, text_color = tracker.process_frame(image)
        
        # સ્ટેટસને વિડીયો ફીડ પર દેખાડવું
        cv2.putText(processed_img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.imshow('SmartAssess AI - Clean Vision Module', processed_img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()