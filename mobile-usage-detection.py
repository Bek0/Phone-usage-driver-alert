import tempfile
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import math
from datetime import datetime, timedelta
import threading
import re

alert_lock = threading.Lock()
count_lock = threading.Lock()

COLORS_RED = sv.ColorPalette.from_hex(["#000000", "#DF0000"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS_RED.by_idx(0), opacity=0.15)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLORS_RED, thickness=1)

def calculate_distance(point1, point2):
    """
    This function calculates the Euclidean distance between two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
    cv2.line(frame, (int(wrist[0]), int(wrist[1])), (int(phone_center[0]), int(phone_center[1])), (0, 0, 0), 2)

    # Display the calculated distance
    cv2.putText(frame, f"{int(distance)}", center_between_phone_and_wrist,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw a rectangle around the phone
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    # Add text alert for phone usage
    cv2.putText(frame, "Phone Near Hand!",
                (top_left[0], top_left[1] - 10),  # مكان النص فوق المربع
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,  # الخط وحجم النص
                (0, 0, 255), 2)

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
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(frame, (int(ear[0]), int(ear[1])), radius=5, color=(255, 255, 255), thickness=-1)

    # Draw lines between ear, nose, and phone center
    cv2.line(frame, (int(ear[0]), int(ear[1])), (int(nose[0]), int(nose[1])), (0, 100, 255), 2)
    cv2.line(frame, (int(nose[0]), int(nose[1])), (int(phone_center[0]), int(phone_center[1])), (0, 100, 255), 2)

    # Calculate midpoint between nose and phone center
    mid_x = int((nose[0] + phone_center[0]) / 2)
    mid_y = int((nose[1] + phone_center[1]) / 2)

    # Add phone alignment label at the midpoint
    cv2.putText(frame, "Phone-Front-Face", (mid_x-50, mid_y - 10),  # Slightly above the midpoint
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def calculate_slope(point1, point2):
    """
    This function calculates the slope between two points.
    """
    if point2[0] != point1[0]:  # To avoid division by zero
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        return float('inf')  

def save_to_pdf(frame, person_id, current_time, alert_type, cam):
    """
    Save detection information to a professional PDF report with enhanced layout and larger image.
    """
    sanitized_time = re.sub(r'[^\w\s-]', '-', str(current_time)[:-8]) 
    pdf_path = fr"D:\Desktop\gp\codes\web\images\{alert_type} {sanitized_time}.pdf"
    with alert_lock:
        if not os.path.exists(pdf_path):
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.setFont("Helvetica-Bold", 24)
            c.setFillColorRGB(0, 80/255, 158/255)  
            c.drawString(50, 750, "Detection Report")
            c.setFont("Helvetica", 12)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(50, 735, "Generated by: AI Surveillance System")
            c.drawString(50, 720, "─" * 80) 
            c.save()

        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter  

        c.setFillColorRGB(0, 80/255, 158/255)
        c.rect(0, height-100, width, 100, fill=1)
        c.setFillColorRGB(1, 1, 1)  
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height-60, "Detection Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height-80, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 16)
        y_position = height - 150
        c.drawString(50, y_position, f"Detection Details At {cam}")

        details_box_height = 100
        c.setStrokeColorRGB(0, 80/255, 158/255)
        c.setLineWidth(2)
        c.rect(50, y_position-details_box_height-10, width-100, details_box_height)

        c.setFont("Helvetica", 12)

        details = [
            f"Alert Type: {alert_type}",
            f"Detection Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Person ID: {person_id}"
        ]

        for i, detail in enumerate(details):
            c.drawString(70, y_position-40-(i*20), detail)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_path = tempfile.mktemp(suffix=".jpg")
        with open(frame_path, 'wb') as img_file:
            img_file.write(buffer.tobytes())

        max_width = width * 0.8
        max_height = height * 0.6  
        img_height, img_width, _ = frame.shape

        scale = min(max_width/img_width, max_height/img_height)
        scaled_width = img_width * scale
        scaled_height = img_height * scale

        image_x = (width - scaled_width) / 2
        image_y = y_position - details_box_height - scaled_height - 50

        c.setStrokeColorRGB(0, 80/255, 158/255)
        c.setLineWidth(2)
        c.rect(image_x-5, image_y-5, scaled_width+10, scaled_height+10)

        c.drawImage(frame_path, image_x, image_y, width=scaled_width, height=scaled_height)
        os.remove(frame_path)

        footer_height = 50
        c.setFillColorRGB(0, 80/255, 158/255)
        c.rect(0, 0, width, footer_height, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica", 10)
        c.drawString(50, 20, "Report generated by the AI Surveillance System | Powered by OpenAI")
        c.drawString(width-100, 20, f"Page {c.getPageNumber()}")

        # Save the page
        c.showPage()
        c.save()

        print(f"PDF Report Updated: {current_time} - Person ID {person_id}")
        pass

def process_video_1(video_path, confidence=0.5):
    """
    Process the video to detect phone usage and make report.
    """
    cap = cv2.VideoCapture(video_path)
    last_report_time = None
    frame_id = 0
    # Keypoint names based on YOLO Pose model for pose estimation
    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 2 != 0:
            continue

        original_width = frame.shape[1]
        original_height = frame.shape[0]
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            new_width = 720
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 720
            new_width = int(new_height * aspect_ratio)
        
        frame = cv2.resize(frame, (new_width, new_height))

        results = model_yolo(frame, classes=[0, 67], conf=confidence)[0]
        
        detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.7)

        pose_results = model_pose.predict(frame, conf=confidence, classes=0)
        keypoints = pose_results[0].keypoints.xy.cpu().numpy() if hasattr(pose_results[0], 'keypoints') else []
        phones_detected = [ 
            detection for detection in detections if detection[-1]['class_name'] == 'cell phone'
        ]

        current_time = datetime.now()
        if phones_detected and (last_report_time is None or (current_time - last_report_time) > timedelta(minutes=1)):
            for phone in phones_detected:

                phone_x1, phone_y1, phone_x2, phone_y2 = phone[0]

                left_wrist = keypoints[0][keypoint_names.index("Left Wrist")]
                right_wrist = keypoints[0][keypoint_names.index("Right Wrist")]
                nose = keypoints[0][keypoint_names.index("Nose")]
                right_ear = keypoints[0][keypoint_names.index("Right Ear")]
                left_ear = keypoints[0][keypoint_names.index("Left Ear")]

                phone_center = ((phone_x1 + phone_x2) // 2, (phone_y1 + phone_y2) // 2)
                phone_height = abs(phone_y2 - phone_y1)
                phone_width = abs(phone_x2 - phone_x1)

                distance_left = calculate_distance(phone_center, left_wrist)
                distance_right = calculate_distance(phone_center, right_wrist)

                right_ear_nose_slope = calculate_slope(right_ear, nose)
                left_ear_nose_slope = calculate_slope(left_ear, nose)

                phone_nose_slope = calculate_slope(nose, phone_center)
                current_time = datetime.now()

                if (abs(right_ear_nose_slope - phone_nose_slope) < 0.5 and right_ear.all())or(abs(left_ear_nose_slope - phone_nose_slope) < 0.5 and left_ear.all()):
                    frame = BOX_ANNOTATOR.annotate(
                        scene=frame,
                        detections=detections,
                    )
                    save_to_pdf(frame, "Driver", current_time, "Phone Front Face", "Camera 1")

                elif distance_left < 2 * phone_height or distance_right < 2 * phone_height:
                    frame = BOX_ANNOTATOR.annotate(
                        scene=frame,
                        detections=detections,
                    )
                    save_to_pdf(frame, "Driver", current_time, "Phone Near Hand", "Camera 1")

                # # Mark the phone center as a circle
                # cv2.circle(frame,(int((phone_x1+phone_x2)//2), int((phone_y1 + phone_y2)//2)), radius=5, color=(255,255,255), thickness=-1)  
                # # Add label for the phone center
                # cv2.putText(frame, " Phone center", (int((phone_x1+phone_x2)//2), int((phone_y1 + phone_y2)//2)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # if abs(right_ear_nose_slope - phone_nose_slope) < 0.5 and right_ear.all():  # أو اختر قيمة مناسبة للتقارب
                #     process_ear(frame, right_ear, nose, phone_center, "Right ear")
                #     cv2.putText(frame, "Nose", (int(nose[0]), int(nose[1]) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                #     cv2.circle(frame, (int(nose[0]), int(nose[1])), radius=5, color=(255,255,255), thickness=-1)  # Nose
                #     cv2.imwrite(r"D:\Desktop\test\slop\slope.jpg", frame)

                # elif abs(left_ear_nose_slope - phone_nose_slope) < 0.5 and left_ear.all():  # أو اختر قيمة مناسبة للتقارب
                #     process_ear(frame, left_ear, nose, phone_center, "Left ear")
                #     cv2.putText(frame, "Nose", (int(nose[0]), int(nose[1]) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                #     cv2.circle(frame, (int(nose[0]), int(nose[1])), radius=5, color=(255,255,255), thickness=-1)  # Nose
                #     cv2.imwrite(r"D:\Desktop\test\slop\slope.jpg", frame)

                # if distance_left < 2 * phone_height:
                #     cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), radius=5, color=(255,255,255), thickness=-1)  # Left wrist
                #     cv2.putText(frame, "Left Wrist", (int(left_wrist[0]), int(left_wrist[1]) - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #     annotate_phone_usage(frame, left_wrist, phone_center, distance_left, phone_width, phone_height, "Left")
                #     cv2.imwrite(r"D:\Desktop\test\slop\diste.jpg", frame)

                # elif distance_right < 2 * phone_height:
                #     cv2.putText(frame, "Right Wrist", (int(right_wrist[0]), int(right_wrist[1]) - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #     cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), radius=5, color=(255,255,255), thickness=-1)  # Right wrist
                #     annotate_phone_usage(frame, right_wrist, phone_center, distance_right, phone_width, phone_height, "Right")
                #     cv2.imwrite(r"D:\Desktop\test\slop\diste.jpg", frame)

            last_report_time = current_time
        cv2.imshow("Person and Face Tracking", frame)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()

model_yolo = None
model_pose = None

def initialize_models():
    global model_yolo, model_pose
    model_yolo = YOLO(r"D:/Desktop/gp/codes/object-detect-yolov8/v8/yolov8m.pt")
    model_pose = YOLO(r"D:/Desktop/gp/codes/object-detect-yolov8/v8/yolov8m-pose.pt")

def start_1():
    global model_yolo, model_pose
    if model_yolo is None or model_pose is None:
        initialize_models()
    process_video_1(video_path_1)

if __name__ == "__main__":

    video_path_1 = r"D:\Desktop\test\slop\istockphoto-1242377929-640_adpp_is.mp4"

    initialize_models()

    thread_1 = threading.Thread(target=start_1)

    thread_1.start()

    thread_1.join()
