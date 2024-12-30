from ultralytics import YOLO
import tempfile
import streamlit as st
import supervision as sv  # Import supervision for tracking
import numpy as np
import cv2
import math  # لاستعمال دالة المسافة الإقليدية

# دالة لحساب المسافة الإقليدية بين نقطتين
def calculate_distance(point1, point2):
    """
    This function calculates the Euclidean distance between two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# دالة لحساب الميل بين نقطتين
def calculate_slope(point1, point2):
    """
    This function calculates the slope between two points.
    """
    if point2[0] != point1[0]:  # To avoid division by zero
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        return float('inf')  # Return infinity if the line is vertical

def process_ear(frame, ear, nose, phone_center,label):
    """
    Function to process ear alignment with the phone and draw necessary annotations.

    Args:
        frame (ndarray): Video frame to modify.
        ear (tuple): Coordinates (x, y) of the ear (either left or right).
        nose (tuple): Coordinates (x, y) of the nose.
        phone_center (tuple): Coordinates (x, y) of the phone's center.
        label (str): Text label to indicate ear alignment.

    Returns:
        None: Modifies the frame in place.
    """
    # Draw ear label
    cv2.putText(frame, label, (int(ear[0]), int(ear[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.circle(frame, (int(ear[0]), int(ear[1])), radius=5, color=(255, 255, 255), thickness=-1)

    # Draw lines between ear, nose, and phone center
    cv2.line(frame, (int(ear[0]), int(ear[1])), (int(nose[0]), int(nose[1])), (0, 100, 255), 2)
    cv2.line(frame, (int(nose[0]), int(nose[1])), (int(phone_center[0]), int(phone_center[1])), (0, 100, 255), 2)

    # Calculate midpoint between nose and phone center
    mid_x = int((nose[0] + phone_center[0]) / 2)
    mid_y = int((nose[1] + phone_center[1]) / 2)

    # Add phone alignment label at the midpoint
    cv2.putText(frame, "Phone-Front-Face", (mid_x-50, mid_y - 10),  # Slightly above the midpoint
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
def annotate_phone_usage(frame, wrist, phone_center, distance, phone_width, phone_height, label):
    """
    Function to annotate phone usage near a wrist.

    Args:
        frame (ndarray): Video frame to modify.
        wrist (tuple): Coordinates (x, y) of the wrist (either left or right).
        phone_center (tuple): Coordinates (x, y) of the phone's center.
        distance (float): Distance between the wrist and phone center.
        phone_width (int): Width of the phone.
        phone_height (int): Height of the phone.
        label (str): Label for the wrist ("Left" or "Right").

    Returns:
        None: Modifies the frame in place.
    """
    # Calculate dimensions for the rectangle to annotate the phone usage
    square_width = 2 * phone_width
    square_height = 2 * phone_height

    # Find the center between the phone and the wrist
    center_between_phone_and_wrist = (
        int((phone_center[0] + wrist[0]) // 2),
        int((phone_center[1] + wrist[1]) // 2)
    )
    top_left = (
        int(center_between_phone_and_wrist[0] - square_width // 2),
        int(center_between_phone_and_wrist[1] - square_height // 2)
    )
    bottom_right = (
        int(center_between_phone_and_wrist[0] + square_width // 2),
        int(center_between_phone_and_wrist[1] + square_height // 2)
    )

    # Draw a line between wrist and phone center
    cv2.line(frame, (int(wrist[0]), int(wrist[1])), (int(phone_center[0]), int(phone_center[1])), (0, 255, 255), 2)

    # Display the calculated distance
    cv2.putText(frame, f"{int(distance)}", center_between_phone_and_wrist,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw a rectangle around the phone
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    # Add text alert for phone usage
    cv2.putText(frame, "Phone Usage!",
                (top_left[0], top_left[1] - 10),  # مكان النص فوق المربع
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,  # الخط وحجم النص
                (0, 0, 255), 2)

# Initialize the YOLO model for object detection (general model)
model = YOLO(r"D:/Desktop/gp/codes/object-detect-yolov8/v8/yolov8m.pt")

# Initialize the YOLO model for pose detection (using pose estimation)
yolo_pose_model = YOLO(r"D:/Desktop/gp/codes/object-detect-yolov8/v8/yolov8m-pose.pt")

# Define color palette for annotations
COLORS = sv.ColorPalette.from_hex([
    "#a7b6eb",  # Coral
    "#D4F1F4"  # Light Cyan
])

# Initialize annotators for drawing bounding boxes and labels
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLORS, thickness=1)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000"), text_padding=5, text_scale=0.25, text_thickness=1
)

# Streamlit interface for video upload
st.title("YOLO with DeepSORT Tracking")
st.write("Upload a video to perform object detection and tracking.")

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Confidence threshold slider to set confidence for detections
confidence = st.slider("Set Confidence Threshold", 0.0, 1.00, 0.5)

# Placeholder for video frames in Streamlit UI
stframe = st.empty()

# Keypoint names based on YOLO Pose model for pose estimation
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Process video if uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Generate video frames from the uploaded file
    generator = sv.get_video_frames_generator(tfile.name)

    # Frame counter to keep track of the frame number
    frame_id = 0

    # Loop through frames in the video
    for frame in generator:
        frame_id += 1
            
        # تصغير حجم الإطار بنسبة معينة (مثلاً 50% من الحجم الأصلي)
        scale_percent = 90  # تغيير حجم الصورة إلى 50% من الحجم الأصلي
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # إعادة تحجيم الإطار
        frame = cv2.resize(frame, dim)
        # Perform inference with YOLO model (object detection)
        results = model(frame, classes=[0, 67], conf=confidence)[0]
        # Convert YOLOv8 results to detections
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms(threshold=0.1)  # Apply Non-Maximum Suppression (NMS)

        # Perform pose tracking using YOLO Pose model
        pose_results = yolo_pose_model.predict(frame, conf=confidence, classes=0)
        keypoints = pose_results[0].keypoints.xy.cpu().numpy() if hasattr(pose_results[0], 'keypoints') else []

        # Create a list of detection labels with confidence values
        labels = [
            f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
            for i in range(len(detections))
        ]

        # Filter detections to check for phones
        phones_detected = [ 
            detection for detection in detections if detection[-1]['class_name'] == 'cell phone'
        ]
        # Initialize an empty message for alerts
        alert_message = ""
        frame = LABEL_ANNOTATOR.annotate(
            scene=frame, detections=detections, labels=labels)
        
        frame = BOX_ANNOTATOR.annotate(
            scene=frame,
            detections=detections,
        )
        # Analyze detected results to check if a phone is near a hand
        for phone in phones_detected:
            # Get the phone bounding box coordinates (x1, y1, x2, y2)
            phone_x1, phone_y1, phone_x2, phone_y2 = phone[0]

            # Extract keypoints for hands (left and right wrists) and nose
            left_wrist = keypoints[0][keypoint_names.index("Left Wrist")]
            right_wrist = keypoints[0][keypoint_names.index("Right Wrist")]
            nose = keypoints[0][keypoint_names.index("Nose")]
            right_ear = keypoints[0][keypoint_names.index("Right Ear")]
            left_ear = keypoints[0][keypoint_names.index("Left Ear")]
            print(right_ear,left_ear)
            # Calculate the phone center and dimensions (height and width)
            phone_center = ((phone_x1 + phone_x2) // 2, (phone_y1 + phone_y2) // 2)
            phone_height = abs(phone_y2 - phone_y1)
            phone_width = abs(phone_x2 - phone_x1)

            # Draw keypoints as circles for visualization
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), radius=5, color=(255,255,255), thickness=-1)  # Left wrist
            cv2.putText(frame, "Left Wrist", (int(left_wrist[0]), int(left_wrist[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.putText(frame, "Right Wrist", (int(right_wrist[0]), int(right_wrist[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), radius=5, color=(255,255,255), thickness=-1)  # Right wrist

            cv2.putText(frame, "Nose", (int(nose[0]), int(nose[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.circle(frame, (int(nose[0]), int(nose[1])), radius=5, color=(255,255,255), thickness=-1)  # Nose

            # Mark the phone center as a circle
            cv2.circle(frame,(int((phone_x1+phone_x2)//2), int((phone_y1 + phone_y2)//2)), radius=5, color=(255,255,255), thickness=-1)  
            # Add label for the phone center
            cv2.putText(frame, "Center", (int((phone_x1+phone_x2)//2), int((phone_y1 + phone_y2)//2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate distances from phone center to both wrists
            distance_left = calculate_distance(phone_center, left_wrist)
            distance_right = calculate_distance(phone_center, right_wrist)

            right_ear_nose_slope = calculate_slope(right_ear, nose)
            left_ear_nose_slope = calculate_slope(left_ear, nose)
            # حساب الميل بين مركز الهاتف والانف
            phone_nose_slope = calculate_slope(nose, phone_center)

            # قارن الميلين إذا كانا متقاربين
            print("right:",abs(right_ear_nose_slope - phone_nose_slope))
            print("left:",abs(left_ear_nose_slope - phone_nose_slope))

            if abs(right_ear_nose_slope - phone_nose_slope) < 0.5 and right_ear.all():  # أو اختر قيمة مناسبة للتقارب
                process_ear(frame, right_ear, nose, phone_center, "Right")
            
            elif abs(left_ear_nose_slope - phone_nose_slope) < 0.5 and left_ear.all():  # أو اختر قيمة مناسبة للتقارب
                process_ear(frame, left_ear, nose, phone_center, "Left")

            # Main condition to check distances and call the function
            if distance_left < 2 * phone_height or distance_right < 2 * phone_height:
                alert_message = "Phone detected near hand!"

                if distance_left < 2 * phone_height:
                    annotate_phone_usage(frame, left_wrist, phone_center, distance_left, phone_width, phone_height, "Left")

                elif distance_right < 2 * phone_height:
                    annotate_phone_usage(frame, right_wrist, phone_center, distance_right, phone_width, phone_height, "Right")

        # Display alert message if necessary
        if alert_message:
            st.warning(alert_message)

        # Display the current frame in the Streamlit UI
        stframe.image(frame, channels="BGR", use_column_width=True)

    # Release video capture
else:
    st.write("Please upload a video file to begin.")
