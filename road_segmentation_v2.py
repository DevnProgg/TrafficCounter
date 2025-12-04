"""
Road Segmentation using Deep Learning (Segformer on Cityscapes)
Production-ready semantic segmentation for road detection.

Author: [Your Name]
Version: 2.0.0
License: MIT
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys
import io
from dataclasses import dataclass
from enum import Enum

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ModelSize(Enum):
    """Available model sizes with their specifications."""
    SMALL = ("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "3.7M params, fastest")
    MEDIUM = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", "24M params, balanced")
    LARGE = ("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "82M params, best quality")


@dataclass
class RoadMetadata:
    """Structured metadata for road detection results."""
    road_angle: float
    center: Tuple[int, int]
    perpendicular_angle: float
    road_detected: bool
    road_coverage: float
    confidence: float = 1.0


class RoadSegmentationError(Exception):
    """Base exception for road segmentation errors."""
    pass


class ModelLoadError(RoadSegmentationError):
    """Raised when model fails to load."""
    pass


class SegmentationError(RoadSegmentationError):
    """Raised when segmentation fails."""
    pass


def configure_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
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

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = configure_logging(Path('models/road_segmentation.log'))


class RoadSegmentationProcessor:
    """Process video frames to detect roads using deep learning."""

    # Cityscapes dataset class IDs
    ROAD_CLASS_ID = 0
    SIDEWALK_CLASS_ID = 1
    MIN_ROAD_PIXELS = 100
    MORPHOLOGY_KERNEL_SIZE = 5

    def __init__(
            self,
            model_size: ModelSize = ModelSize.SMALL,
            device: Optional[str] = None,
            cache_dir: Optional[Path] = None
    ):
        """
        Initialize the road segmentation processor.

        Args:
            model_size: Model size to use
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model files

        Raises:
            ModelLoadError: If model fails to load
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_size.value[0]
        self.logger.info(f"Initializing with {model_size.name} model: {model_size.value[1]}")

        self._validate_dependencies()
        self._initialize_device(device)
        self._load_model(cache_dir)

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are installed."""
        try:
            import torch
            import transformers
            from PIL import Image
        except ImportError as e:
            raise ModelLoadError(
                "Required libraries not installed. "
                "Install with: pip install transformers torch torchvision Pillow"
            ) from e

    def _initialize_device(self, device: Optional[str]) -> None:
        """Initialize compute device."""
        import torch

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Using device: {self.device}")

        if self.device == "cuda":
            import torch
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def _load_model(self, cache_dir: Optional[Path]) -> None:
        """Load the segmentation model."""
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            import torch

            self.logger.info("Loading model (may take a minute on first run)...")

            kwargs = {}
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs['cache_dir'] = str(cache_dir)

            self.processor = SegformerImageProcessor.from_pretrained(
                self.model_name, **kwargs
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name, **kwargs
            )

            self.model.to(self.device)
            self.model.eval()

            # Disable gradient computation for inference
            for param in self.model.parameters():
                param.requires_grad = False

            self.logger.info("Model loaded successfully")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation on image.

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            Segmentation map (H, W) with class IDs

        Raises:
            SegmentationError: If segmentation fails
        """
        if image is None or image.size == 0:
            raise SegmentationError("Invalid input image")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise SegmentationError(f"Expected 3-channel image, got shape {image.shape}")

        try:
            import torch
            from PIL import Image

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Preprocess
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Upsample logits to original image size
            logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False
            )

            # Get class predictions
            seg_map = logits.argmax(dim=1)[0].cpu().numpy()

            return seg_map

        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {e}") from e

    def extract_road_mask(
            self,
            seg_map: np.ndarray,
            include_sidewalk: bool = False
    ) -> np.ndarray:
        """
        Extract road mask from segmentation map.

        Args:
            seg_map: Segmentation map with class IDs
            include_sidewalk: Whether to include sidewalk as part of road

        Returns:
            Binary mask of road (uint8, values 0 or 255)
        """
        # Create mask for road class
        road_mask = (seg_map == self.ROAD_CLASS_ID).astype(np.uint8) * 255

        if include_sidewalk:
            sidewalk_mask = (seg_map == self.SIDEWALK_CLASS_ID).astype(np.uint8) * 255
            road_mask = cv2.bitwise_or(road_mask, sidewalk_mask)

        # Clean up mask with morphological operations
        kernel = np.ones((self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        return road_mask

    def calculate_road_orientation(
            self,
            mask: np.ndarray
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Calculate the orientation and center of the road from mask.

        Args:
            mask: Binary mask of the road

        Returns:
            Tuple of (angle in degrees, center point)

        Raises:
            ValueError: If no valid road contours found
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No contours found in mask")

        # Get the largest contour (main road)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < self.MIN_ROAD_PIXELS:
            raise ValueError("Road area too small")

        # Get moments to find center
        M = cv2.moments(largest_contour)

        if M['m00'] == 0:
            # Fallback to image center
            h, w = mask.shape[:2]
            center = (w // 2, h // 2)
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)

        # Calculate orientation
        angle = self._calculate_contour_angle(largest_contour)

        self.logger.debug(f"Road orientation: {angle:.2f}°, Center: {center}")
        return angle, center

    def _calculate_contour_angle(self, contour: np.ndarray) -> float:
        """Calculate angle of contour using ellipse fitting or PCA."""
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                return ellipse[2]
            except cv2.error:
                pass

        # Fallback to PCA
        points = contour.reshape(-1, 2).astype(np.float64)
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        return angle

    def get_perpendicular_line(
            self,
            image: np.ndarray,
            angle: float,
            center: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate perpendicular line coordinates.

        Args:
            image: Input image
            angle: Road angle in degrees
            center: Center point for the line

        Returns:
            Tuple of (x1, y1, x2, y2) line coordinates
        """
        # Calculate perpendicular angle (horizontal line)
        perpendicular_angle = 180.0  # Horizontal line

        # Convert to radians
        rad = np.deg2rad(perpendicular_angle)

        # Calculate line endpoints
        h, w = image.shape[:2]
        length = max(w, h)

        x1 = int(center[0] - length * np.cos(rad))
        y1 = int(center[1] - length * np.sin(rad))
        x2 = int(center[0] + length * np.cos(rad))
        y2 = int(center[1] + length * np.sin(rad))

        return x1, y1, x2, y2

    def draw_perpendicular_line(
            self,
            image: np.ndarray,
            angle: float,
            center: Tuple[int, int],
            color: Tuple[int, int, int] = (0, 255, 0),
            thickness: int = 3
    ) -> np.ndarray:
        """
        Draw a line perpendicular to the road orientation.

        Args:
            image: Input image
            angle: Road angle in degrees
            center: Center point for the line
            color: Line color in BGR
            thickness: Line thickness

        Returns:
            Image with perpendicular line drawn
        """
        output = image.copy()
        x1, y1, x2, y2 = self.get_perpendicular_line(image, angle, center)

        # Draw the line
        cv2.line(output, (x1, y1), (x2, y2), color, thickness)

        # Draw center point
        cv2.circle(output, center, 8, (0, 0, 255), -1)

        return output

    def process_frame(
            self,
            frame: np.ndarray,
            visualize: bool = True,
            include_sidewalk: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[RoadMetadata]]:
        """
        Process a single frame to detect road and draw perpendicular line.

        Args:
            frame: Input video frame (BGR format)
            visualize: Whether to draw visualizations
            include_sidewalk: Include sidewalk in road detection

        Returns:
            Tuple of (processed frame, road mask, metadata)
        """
        try:
            self.logger.debug("Processing frame...")

            # Run semantic segmentation
            seg_map = self.segment_image(frame)

            # Extract road mask
            road_mask = self.extract_road_mask(seg_map, include_sidewalk)

            # Check if road was detected
            road_pixels = np.sum(road_mask > 0)
            if road_pixels < self.MIN_ROAD_PIXELS:
                self.logger.warning(f"Insufficient road pixels detected: {road_pixels}")
                return frame, None, None

            # Calculate road orientation
            angle, center = self.calculate_road_orientation(road_mask)

            # Create visualization
            output_frame = frame.copy()

            if visualize:
                output_frame = self._create_visualization(
                    output_frame, road_mask, angle, center
                )

            # Prepare metadata
            total_pixels = road_mask.shape[0] * road_mask.shape[1]
            metadata = RoadMetadata(
                road_angle=float(angle),
                center=center,
                perpendicular_angle=float(angle + 90),
                road_detected=True,
                road_coverage=float((road_pixels / total_pixels) * 100)
            )

            self.logger.debug("Frame processed successfully")
            return output_frame, road_mask, metadata

        except (SegmentationError, ValueError) as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, None, None

    def _create_visualization(
            self,
            frame: np.ndarray,
            mask: np.ndarray,
            angle: float,
            center: Tuple[int, int]
    ) -> np.ndarray:
        """Create visualization overlay on frame."""
        # Draw road mask overlay
        mask_overlay = np.zeros_like(frame)
        mask_overlay[mask > 0] = [0, 255, 255]  # Yellow for road
        output = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

        # Draw perpendicular counting line
        output = self.draw_perpendicular_line(output, angle, center)

        # Calculate statistics
        road_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = (road_pixels / total_pixels) * 100

        # Add text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 2

        texts = [
            (f"Road Angle: {angle:.1f}°", (10, 30)),
            (f"Counting Line (Perpendicular)", (10, 60)),
            (f"Road Coverage: {coverage:.1f}%", (10, 90))
        ]

        for text, pos in texts:
            cv2.putText(output, text, pos, font, font_scale, font_color, thickness)

        return output


def process_image_file(
        image_path: Path,
        processor: RoadSegmentationProcessor,
        include_sidewalk: bool = False,
        show_result: bool = True
) -> Optional[RoadMetadata]:
    """
    Process a single image file.

    Args:
        image_path: Path to image file
        processor: Road segmentation processor
        include_sidewalk: Include sidewalk in detection
        show_result: Display result in window

    Returns:
        Metadata if successful, None otherwise
    """
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    # Read image
    frame = cv2.imread(str(image_path))

    if frame is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    logger.info(f"Processing: {image_path.name} (size: {frame.shape})")

    # Process frame
    output, mask, metadata = processor.process_frame(
        frame,
        visualize=True,
        include_sidewalk=include_sidewalk
    )

    # Display results
    if metadata:
        logger.info(f"Road detected - Coverage: {metadata.road_coverage:.1f}%, "
                    f"Angle: {metadata.road_angle:.1f}°")

        if show_result:
            cv2.imshow('Road Detection', output)
            if mask is not None:
                cv2.imshow('Road Mask', mask)
            logger.info("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        logger.warning("No road detected in the image")
        if show_result:
            cv2.imshow('Original', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return metadata


def batch_process_images(
        image_dir: Path,
        model_size: ModelSize = ModelSize.SMALL,
        include_sidewalk: bool = False,
        output_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Process all images in a directory.

    Args:
        image_dir: Directory containing images
        model_size: Model size to use
        include_sidewalk: Include sidewalk in detection
        output_dir: Optional directory to save results

    Returns:
        List of results for each image
    """
    if not image_dir.exists():
        logger.error(f"Directory not found: {image_dir}")
        return []

    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in extensions]

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return []

    logger.info(f"Found {len(image_files)} images to process")

    # Initialize processor once
    processor = RoadSegmentationProcessor(model_size=model_size)

    results = []

    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] Processing {image_path.name}")

        metadata = process_image_file(
            image_path,
            processor,
            include_sidewalk=include_sidewalk,
            show_result=False
        )

        result = {
            'filename': image_path.name,
            'path': str(image_path),
            'metadata': metadata
        }
        results.append(result)

        # Save output if requested
        if output_dir and metadata:
            output_dir.mkdir(parents=True, exist_ok=True)
            frame = cv2.imread(str(image_path))
            output, _, _ = processor.process_frame(frame, visualize=True)
            output_path = output_dir / f"processed_{image_path.name}"
            cv2.imwrite(str(output_path), output)

    # Summary
    successful = sum(1 for r in results if r['metadata'] is not None)
    logger.info(f"Processing complete: {successful}/{len(image_files)} images successful")

    return results


def main():
    """Main entry point for the application."""
    logger.info("=" * 70)
    logger.info("Road Segmentation Application v2.0.0")
    logger.info("=" * 70)

    # Check dependencies
    try:
        import torch
        from transformers import __version__ as transformers_version
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers_version}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error("Required libraries not installed!")
        logger.error("Install with: pip install transformers torch torchvision Pillow")
        logger.error("For GPU: pip install torch torchvision --index-url "
                     "https://download.pytorch.org/whl/cu118")
        return

    # Example usage
    image_dir = Path("./roadimages")

    if not image_dir.exists():
        logger.warning(f"Image directory not found: {image_dir}")
        logger.info("Please create './roadimages' directory and add test images")
        return

    # Batch process images
    results = batch_process_images(
        image_dir,
        model_size=ModelSize.SMALL,
        include_sidewalk=False,
        output_dir=Path("./output")
    )

    logger.info("=" * 70)
    logger.info("Processing complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)