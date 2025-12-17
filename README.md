# Traffic Analysis and Counting System

This project is a comprehensive traffic monitoring and analysis system that uses computer vision and deep learning to detect, track, and count vehicles in video footage. It automatically identifies road surfaces, counts vehicles as they cross a designated line, and generates detailed analytics reports.

## Features

-   **Automatic Road Segmentation:** Uses a Segformer deep learning model to automatically detect the road area in a video frame.
-   **Dynamic Counting Line:** Automatically places a counting line perpendicular to the direction of the road.
-   **Vehicle Detection and Tracking:** Employs a YOLO (You Only Look Once) model to detect and track multiple vehicle types (cars, trucks, buses, etc.).
-   **Traffic Counting:** Counts vehicles as they cross the virtual line.
-   **Data Analytics:** Generates detailed analytics from the collected traffic data, including:
    -   Vehicle counts by class.
    -   Traffic flow rate (vehicles per minute).
    -   Vehicle composition pie/bar charts.
    -   Time-series analysis of traffic patterns.
-   **Visualization:** Overlays all detections, tracking information, and statistics directly onto the output video.

## How It Works

The system processes video files in a three-stage pipeline:

1.  **Road Segmentation (`road_segmentation_v2.py`):**
    -   A frame is extracted from the video.
    -   A pre-trained Segformer model segments the image to identify the road surface.
    -   The orientation of the road is calculated, and a perpendicular line is determined to be used as the counting line.

2.  **Detection and Tracking (`road_detection.py`):**
    -   The video is processed frame by frame.
    -   The YOLO model detects and tracks vehicles in each frame.
    -   The system checks if a tracked vehicle's center point has crossed the counting line.
    -   If a vehicle crosses the line, it is counted, and its class (car, truck, etc.) is recorded.
    -   The output video is saved with all visual information overlaid, and a JSON file containing the traffic statistics is generated.

3.  **Analytics (`analytics.py`):**
    -   The JSON statistics file is read.
    -   The script generates various plots and dashboards using Matplotlib and Seaborn.
    -   The visualizations are saved as PNG images for reporting.

## Directory Structure

```
.
├── analytics.py            # Generates analytics reports from statistics files.
├── road_detection.py       # Main script for vehicle detection and counting.
├── road_segmentation_v2.py # Script for road segmentation.
├── models/                 # Directory for storing ML models.
│   └── yolo11l.pt          # Example YOLO model.
├── output/                 # Directory for all output files.
│   ├── analytics/          # Output for analytics charts.
│   ├── statistics/         # Output for JSON statistics files.
│   └── processed_vid1.mp4  # Example processed video.
├── roadimages/             # Directory for input images for road segmentation.
└── trafficvideos/          # Directory for input traffic videos.
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required Python libraries:**
    The primary dependencies are listed in the files. You can install them using pip:
    ```bash
    pip install ultralytics opencv-python numpy matplotlib seaborn transformers torch torchvision Pillow
    ```
    *Note: For GPU support with PyTorch, follow the installation instructions on the official [PyTorch website](https://pytorch.org/get-started/locally/).*

4.  **Download Models:**
    -   The YOLO model (`yolo11l.pt`) should be placed in the `models/` directory.
    -   The Segformer model will be downloaded automatically by the `transformers` library on its first run and cached.

## Usage

There are three main scripts you can run:

### 1. Road Segmentation (Optional)

To test the road segmentation on a set of images:

1.  Place your images in the `roadimages/` directory.
2.  Run the script:
    ```bash
    python road_segmentation_v2.py
    ```
    Processed images will be saved in the `output/` directory.

### 2. Traffic Detection and Counting

To process a traffic video:

1.  Place your video file in the `trafficvideos/` directory.
2.  Update the `video_path` in the `main()` function of `road_detection.py` to point to your video.
3.  Run the script:
    ```bash
    python road_detection.py
    ```
    -   A window will pop up displaying the processed video in real-time.
    -   The final video will be saved to the `output/` directory.
    -   A JSON statistics file will be saved in `output/statistics/`.

### 3. Generate Analytics

To generate analytics from the traffic statistics:

1.  Ensure you have one or more `*.json` statistics files in the `output/statistics/` directory.
2.  Run the script:
    ```bash
    python analytics.py
    ```
    -   The script will generate PNG images of the charts in the `output/analytics/` directory.

## Dependencies

-   **Core:**
    -   `opencv-python`
    -   `numpy`
-   **Machine Learning:**
    -   `ultralytics`: For YOLO object detection.
    -   `torch` & `torchvision`: For the deep learning models.
    -   `transformers`: For the Segformer segmentation model.
    -   `Pillow`
-   **Analytics:**
    -   `matplotlib`
    -   `seaborn`
