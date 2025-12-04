"""
Traffic Detection and Counting System
Production-ready traffic monitoring with YOLO tracking and road segmentation.

Author: [Your Name]
Version: 2.0.0
License: MIT
"""
import cv2
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Set, List
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO not installed. Install with: pip install ultralytics"
    )

from road_segmentation_v2 import RoadSegmentationProcessor


@dataclass
class TrafficStats:
    """Container for traffic statistics."""
    class_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    crossed_ids: Set[int] = field(default_factory=set)
    total_detections: int = 0
    frames_processed: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert stats to dictionary for serialization."""
        return {
            'class_counts': dict(self.class_counts),
            'total_vehicles': sum(self.class_counts.values()),
            'frames_processed': self.frames_processed,
            'unique_objects_tracked': len(self.crossed_ids),
            'start_time': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class TrafficDetectionError(Exception):
    """Base exception for traffic detection errors."""
    pass


class VideoLoadError(TrafficDetectionError):
    """Raised when video fails to load."""
    pass


class ModelInitError(TrafficDetectionError):
    """Raised when model initialization fails."""
    pass


def configure_logging(
        log_file: Optional[Path] = None,
        level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = configure_logging(Path('models/road_detection.log'))


class RoadDetectionProcessor:
    """
    Traffic detection and counting processor using YOLO and road segmentation.
    """

    # Default COCO classes for vehicle detection
    DEFAULT_VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        6: 'train',
        7: 'truck'
    }

    # Visual settings
    COLORS = {
        'line': (0, 0, 255),  # Red for counting line
        'box': (0, 255, 0),  # Green for bounding boxes
        'center': (0, 0, 255),  # Red for center points
        'text': (255, 255, 255),  # White for text
        'text_bg': (0, 0, 0)  # Black for text background
    }

    MIN_ROAD_PIXELS = 100
    LINE_THICKNESS = 3
    BOX_THICKNESS = 2

    def __init__(
            self,
            model_path: Path = Path('./models/yolo11l.pt'),
            classes: Optional[List[int]] = None,
            confidence_threshold: float = 0.5,
            enable_road_segmentation: bool = True
    ):
        """
        Initialize traffic detection processor.

        Args:
            model_path: Path to YOLO model weights
            classes: List of class IDs to detect (None for all)
            confidence_threshold: Minimum confidence for detections
            enable_road_segmentation: Whether to use road segmentation

        Raises:
            ModelInitError: If model initialization fails
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("=" * 70)
        self.logger.info("Initializing Traffic Detection System")
        self.logger.info("=" * 70)

        self.confidence_threshold = confidence_threshold
        self.enable_road_segmentation = enable_road_segmentation

        # Initialize tracking variables
        self.stats = TrafficStats()
        self.counting_line: Optional[Tuple[int, int, int, int]] = None
        self.is_road_segmented = False

        # Load YOLO model
        self._load_yolo_model(model_path, classes)

        # Initialize road segmentation if enabled
        if enable_road_segmentation:
            self._initialize_road_segmentation()

        self.logger.info("Initialization complete")

    def _load_yolo_model(
            self,
            model_path: Path,
            classes: Optional[List[int]]
    ) -> None:
        """Load and validate YOLO model."""
        if not model_path.exists():
            raise ModelInitError(f"Model file not found: {model_path}")

        try:
            self.logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(str(model_path))
            self.class_list = self.model.names

            # Set detection classes
            if classes is None:
                self.detection_classes = list(self.DEFAULT_VEHICLE_CLASSES.keys())
            else:
                self.detection_classes = classes

            class_names = [self.class_list.get(i, f"Class_{i}")
                           for i in self.detection_classes]
            self.logger.info(f"Detecting classes: {class_names}")
            self.logger.info(f"Confidence threshold: {self.confidence_threshold}")

        except Exception as e:
            raise ModelInitError(f"Failed to load YOLO model: {e}") from e

    def _initialize_road_segmentation(self) -> None:
        """Initialize road segmentation processor."""
        try:
            self.logger.info("Initializing road segmentation processor...")
            self.segmentation_processor = RoadSegmentationProcessor()
            self.logger.info("Road segmentation initialized")
        except Exception as e:
            self.logger.error(f"Road segmentation initialization failed: {e}")
            self.logger.warning("Continuing without road segmentation")
            self.enable_road_segmentation = False

    def _segment_road_and_set_line(self, frame: np.ndarray) -> bool:
        """
        Perform road segmentation and set counting line.

        Args:
            frame: Video frame

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Performing road segmentation...")

            # Run semantic segmentation
            seg_map = self.segmentation_processor.segment_image(frame)
            road_mask = self.segmentation_processor.extract_road_mask(seg_map)

            # Check if road was detected
            road_pixels = np.sum(road_mask > 0)
            if road_pixels < self.MIN_ROAD_PIXELS:
                self.logger.warning(f"Insufficient road pixels: {road_pixels}")
                return False

            # Calculate road orientation and get counting line
            angle, center = self.segmentation_processor.calculate_road_orientation(road_mask)
            self.counting_line = self.segmentation_processor.get_perpendicular_line(frame, angle, center)

            self.logger.info(f"Road segmented - Angle: {angle:.1f}°, Center: {center}")
            self.logger.info(f"Counting line set: {self.counting_line}")

            return True

        except Exception as e:
            self.logger.error(f"Road segmentation failed: {e}")
            return False

    def _set_default_counting_line(self, frame: np.ndarray) -> None:
        """Set a default horizontal counting line at 60% height."""
        h, w = frame.shape[:2]
        y_position = int(h * 0.6)
        self.counting_line = (0, y_position, w, y_position)
        self.logger.info(f"Using default counting line at y={y_position}")

    def _check_line_crossing(
            self,
            center_x: int,
            center_y: int,
            track_id: int,
            class_name: str
    ) -> bool:
        """
        Check if object has crossed the counting line.

        Args:
            center_x: Object center X coordinate
            center_y: Object center Y coordinate
            track_id: Object tracking ID
            class_name: Object class name

        Returns:
            True if object crossed the line, False otherwise
        """
        if self.counting_line is None:
            return False

        # Check if object crossed line and hasn't been counted yet
        line_y = self.counting_line[1]  # Y coordinate of horizontal line

        if center_y > line_y and track_id not in self.stats.crossed_ids:
            self.stats.crossed_ids.add(track_id)
            self.stats.class_counts[class_name] += 1
            self.logger.info(f"Vehicle crossed: ID={track_id}, Class={class_name}")
            return True

        return False

    def _draw_counting_line(self, frame: np.ndarray) -> None:
        """Draw counting line on frame."""
        if self.counting_line is None:
            return

        x1, y1, x2, y2 = self.counting_line
        cv2.line(frame, (x1, y1), (x2, y2), self.COLORS['line'], self.LINE_THICKNESS)

        # Add label for counting line
        cv2.putText(
            frame,
            "COUNTING LINE",
            (10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLORS['line'],
            2
        )

    def _draw_detections(
            self,
            frame: np.ndarray,
            boxes: np.ndarray,
            track_ids: List[int],
            class_indices: List[int],
            confidences: np.ndarray
    ) -> None:
        """
        Draw bounding boxes and tracking information.

        Args:
            frame: Video frame
            boxes: Bounding boxes (x1, y1, x2, y2)
            track_ids: Tracking IDs
            class_indices: Class indices
            confidences: Detection confidences
        """
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            # Skip low confidence detections
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get class name
            class_name = self.class_list.get(class_idx, f"Class_{class_idx}")

            # Check line crossing
            self._check_line_crossing(cx, cy, track_id, class_name)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS['box'], self.BOX_THICKNESS)

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, self.COLORS['center'], -1)

            # Draw label with background
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                self.COLORS['text_bg'],
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLORS['text'],
                2
            )

    def _draw_statistics(self, frame: np.ndarray) -> None:
        """Draw traffic statistics on frame."""
        h, w = frame.shape[:2]

        # Create semi-transparent background for stats
        overlay = frame.copy()
        panel_height = 40 + len(self.stats.class_counts) * 35
        cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw title
        cv2.putText(
            frame,
            "TRAFFIC STATISTICS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Draw counts by class
        y_offset = 70
        total_count = 0

        for class_name, count in sorted(self.stats.class_counts.items()):
            text = f"{class_name.capitalize()}: {count}"
            cv2.putText(
                frame,
                text,
                (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 35
            total_count += count

        # Draw total
        if total_count > 0:
            cv2.line(frame, (20, y_offset - 20), (340, y_offset - 20), (255, 255, 255), 1)
            cv2.putText(
                frame,
                f"TOTAL: {total_count}",
                (30, y_offset + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    def _draw_info_panel(self, frame: np.ndarray) -> None:
        """Draw information panel with system status."""
        h, w = frame.shape[:2]

        # Bottom panel
        info_texts = [
            f"Frame: {self.stats.frames_processed}",
            f"Active Tracks: {len(self.stats.crossed_ids)}",
            f"Press 'q' to quit | 's' to save stats"
        ]

        y_start = h - 30 * len(info_texts) - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for i, text in enumerate(info_texts):
            cv2.putText(
                frame,
                text,
                (10, y_start + 20 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for vehicle detection and counting.

        Args:
            frame: Input video frame

        Returns:
            Processed frame with visualizations
        """
        self.stats.frames_processed += 1

        # Perform road segmentation on first frame if enabled
        if not self.is_road_segmented:
            self.is_road_segmented = True

            if self.enable_road_segmentation:
                if not self._segment_road_and_set_line(frame):
                    self.logger.warning("Road segmentation failed, using default line")
                    self._set_default_counting_line(frame)
            else:
                self._set_default_counting_line(frame)

        # Run YOLO tracking
        try:
            results = self.model.track(
                frame,
                persist=True,
                classes=self.detection_classes,
                conf=self.confidence_threshold,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"YOLO tracking failed: {e}")
            return frame

        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes

            if boxes_data.id is not None:  # Check if tracking IDs exist
                boxes = boxes_data.xyxy.cpu().numpy()
                track_ids = boxes_data.id.int().cpu().tolist()
                class_indices = boxes_data.cls.cpu().int().tolist()
                confidences = boxes_data.conf.cpu().numpy()

                self.stats.total_detections += len(boxes)

                # Draw counting line
                self._draw_counting_line(frame)

                # Draw detections
                self._draw_detections(frame, boxes, track_ids, class_indices, confidences)

        # Draw statistics and info
        self._draw_statistics(frame)
        self._draw_info_panel(frame)

        return frame

    def process_video(
            self,
            video_path: Path,
            output_path: Optional[Path] = None,
            display: bool = True,
            save_stats: bool = True
    ) -> TrafficStats:
        """
        Process entire video for traffic detection and counting.

        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            display: Whether to display video during processing
            save_stats: Whether to save statistics to JSON

        Returns:
            Traffic statistics

        Raises:
            VideoLoadError: If video cannot be loaded
        """
        if not video_path.exists():
            raise VideoLoadError(f"Video file not found: {video_path}")

        self.logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise VideoLoadError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video info - Resolution: {width}x{height}, "
                         f"FPS: {fps}, Frames: {total_frames}")

        # Setup video writer if output path provided
        writer = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            self.logger.info(f"Saving output to: {output_path}")

        try:
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Process frame
                processed_frame = self.process_frame(frame)

                # Write to output video
                if writer:
                    writer.write(processed_frame)

                # Display frame
                if display:
                    cv2.imshow("Traffic Detection & Counting", processed_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Processing stopped by user")
                        break
                    elif key == ord('s'):
                        self._save_statistics(video_path.stem)

                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        self.logger.info("Processing complete")
        self.logger.info(f"Total vehicles counted: {sum(self.stats.class_counts.values())}")

        # Save statistics
        if save_stats:
            self._save_statistics(video_path.stem)

        return self.stats

    def _save_statistics(self, video_name: str) -> None:
        """Save traffic statistics to JSON file."""
        stats_dir = Path('output/statistics')
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_file = stats_dir / f"stats_{video_name}_{timestamp}.json"

        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)

            self.logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")

    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self.stats = TrafficStats()
        self.is_road_segmented = False
        self.counting_line = None
        self.logger.info("Statistics reset")


def main():
    """Main entry point for the application."""
    logger.info("=" * 70)
    logger.info("Traffic Detection & Counting System v2.0.0")
    logger.info("=" * 70)

    # Configuration
    video_path = Path('./trafficvideos/vid3.mp4')
    output_path = Path('./output/processed_vid1.mp4')
    model_path = Path('./models/yolo11l.pt')

    # Validate paths
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        logger.info("Please ensure video file exists at the specified path")
        return

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please download YOLO model weights")
        return

    try:
        # Initialize processor
        processor = RoadDetectionProcessor(
            model_path=model_path,
            confidence_threshold=0.5,
            enable_road_segmentation=True
        )

        # Process video
        stats = processor.process_video(
            video_path=video_path,
            output_path=output_path,
            display=True,
            save_stats=True
        )

        # Print summary
        logger.info("=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Frames processed: {stats.frames_processed}")
        logger.info(f"Total vehicles: {sum(stats.class_counts.values())}")
        logger.info("Vehicle breakdown:")
        for class_name, count in sorted(stats.class_counts.items()):
            logger.info(f"  {class_name.capitalize()}: {count}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()