# AI Surveillance Detection

## Overview
AI Surveillance Detection is a real-time system designed to enhance security by identifying suspicious or prohibited behaviors, such as using phones while driving or engaging in violent activities, through video surveillance. The project leverages advanced deep learning techniques for object detection, pose estimation, and temporal analysis.

## Features
- **Real-time detection**: Identifies behaviors like phone usage, violent actions, and more in live video feeds.
- **Multi-class detection**: Supports detection of multiple actions within the same framework.
- **Advanced algorithms**: Utilizes YOLO models for object detection and pose estimation for behavior analysis.
- **Automated reporting**: Generates professional PDF reports for each detected incident, including timestamp, location, and image evidence.

## Key Technologies
- **YOLOv8**: Used for object detection and classification.
- **YOLO-Pose**: Applied for pose estimation to analyze human keypoints.
- **Python & OpenCV**: Core programming languages for video processing and analysis.
- **ReportLab**: Used to generate PDF reports of detected activities.

## Project Structure
```plaintext
├── data/                      # Dataset and preprocessed data
├── models/                    # Trained models and YOLO weights
├── src/                       # Source code for the project
│   ├── video_processing.py    # Video processing and detection logic
│   ├── report_generator.py    # Generates detailed PDF reports
│   └── utils/                 # Utility functions
├── output/                    # Generated reports and results
└── README.md                  # Project documentation (this file)
```

## Installation
### Prerequisites
- Python 3.8+
- pip package manager
- GPU with CUDA support (optional, for faster inference)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Surveillance-Detection.git
   cd AI-Surveillance-Detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLO weights:
   - Place the YOLOv8 model weights (e.g., `yolov8n.pt`) in the `models/` directory.

## Usage
### Running the Detection System
1. Update the `video_path` variable in `src/video_processing.py` with your video file or camera feed.
2. Run the script:
   ```bash
   python src/video_processing.py
   ```
3. The system will process the video feed and generate PDF reports for detected incidents in the `output/` folder.

### Customizing Detection
You can modify the detection thresholds, classes, or alert conditions in the configuration file or directly in the source code.

## Examples
### Example PDF Report
Each report contains:
- **Header**: Report title and generation date.
- **Details**: Alert type, person ID, detection time, and location.
- **Annotated Image**: Highlighting detected behavior with bounding boxes.

![Sample Report](./docs/sample_report.png)

## Contribution
Contributions are welcome! Feel free to submit a pull request or open an issue to suggest improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Ultralytics YOLO** for the object detection framework.
- **OpenCV** for image and video processing.
- **ReportLab** for creating professional PDF reports.

---
Start enhancing security with AI Surveillance Detection today!

