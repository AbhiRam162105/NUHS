"""
Active Contour (Snake) Detector

Boundary detection using active contour models (snakes).
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import logging

logger = logging.getLogger(__name__)


class ActiveContourDetector:
    """
    Detects boundaries using active contour models (snakes).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Snake parameters
            'alpha': 0.015,  # Elasticity (continuity)
            'beta': 10,  # Rigidity (smoothness)
            'gamma': 0.001,  # Step size (viscosity)
            'max_iterations': 2000,
            'convergence_threshold': 0.1,

            # Initialization
            'init_method': 'circular',  # 'circular', 'edge_based', 'roi_center'
            'init_radius_ratio': 0.4,  # For circular init (ratio of ROI size)
            'init_num_points': 100,  # Number of points in initial contour

            # Image preprocessing for snake
            'gaussian_blur_sigma': 2.0,

            # Boundary conditions
            'bc': 'periodic',  # 'periodic' or 'fixed'

            # Multi-scale
            'use_multiscale': False,
            'scales': [0.5, 1.0]
        }

        if config:
            self.config.update(config)

        logger.info(f"Active Contour detector initialized with config: {self.config}")

    def detect(self, frame: np.ndarray, initial_contour: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect boundary using active contour.

        Args:
            frame (np.ndarray): Input grayscale frame
            initial_contour (Optional[np.ndarray]): Initial contour points (Nx2)

        Returns:
            np.ndarray: Detected boundary contour (Nx2)
        """
        # Preprocess image for snake
        processed = self._preprocess_for_snake(frame)

        # Initialize contour if not provided
        if initial_contour is None:
            initial_contour = self._initialize_contour(frame)

        # Run active contour
        snake = active_contour(
            processed,
            initial_contour,
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            gamma=self.config['gamma'],
            max_iterations=self.config['max_iterations'],
            convergence=self.config['convergence_threshold'],
            boundary_condition=self.config['bc']
        )

        return snake

    def detect_multiscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Multi-scale active contour detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Detected boundary contour
        """
        scales = self.config['scales']
        h, w = frame.shape

        contour = None

        for scale in scales:
            # Resize frame
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))

            # Initialize or scale previous contour
            if contour is None:
                init_contour = None
            else:
                init_contour = contour * scale

            # Detect
            scaled_contour = self.detect(scaled_frame, init_contour)

            # Scale back to original size
            contour = scaled_contour / scale

        return contour

    def _preprocess_for_snake(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image for active contour.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Preprocessed frame
        """
        # Normalize to [0, 1]
        normalized = frame.astype(float) / 255.0

        # Apply Gaussian blur
        sigma = self.config['gaussian_blur_sigma']
        blurred = gaussian(normalized, sigma=sigma)

        return blurred

    def _initialize_contour(self, frame: np.ndarray) -> np.ndarray:
        """
        Initialize contour for snake.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Initial contour points (Nx2)
        """
        method = self.config['init_method']
        h, w = frame.shape

        if method == 'circular':
            # Circular initialization at center
            cx, cy = w // 2, h // 2
            radius = min(w, h) * self.config['init_radius_ratio']
            return self._create_circular_contour(cx, cy, radius)

        elif method == 'edge_based':
            # Use edge detection for initialization
            edges = cv2.Canny(frame, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Use largest contour
                largest = max(contours, key=cv2.contourArea)
                # Resample to desired number of points
                return self._resample_contour(largest.squeeze(), self.config['init_num_points'])
            else:
                # Fallback to circular
                cx, cy = w // 2, h // 2
                radius = min(w, h) * self.config['init_radius_ratio']
                return self._create_circular_contour(cx, cy, radius)

        elif method == 'roi_center':
            # Ellipse at center
            cx, cy = w // 2, h // 2
            rx = w * 0.4
            ry = h * 0.4
            return self._create_elliptical_contour(cx, cy, rx, ry)

        else:
            # Default to circular
            cx, cy = w // 2, h // 2
            radius = min(w, h) * self.config['init_radius_ratio']
            return self._create_circular_contour(cx, cy, radius)

    def _create_circular_contour(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """
        Create circular contour.

        Args:
            cx, cy (float): Center coordinates
            radius (float): Radius

        Returns:
            np.ndarray: Contour points (Nx2)
        """
        num_points = self.config['init_num_points']
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)

        return np.column_stack([x, y])

    def _create_elliptical_contour(self, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        """
        Create elliptical contour.

        Args:
            cx, cy (float): Center coordinates
            rx, ry (float): Radii

        Returns:
            np.ndarray: Contour points (Nx2)
        """
        num_points = self.config['init_num_points']
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x = cx + rx * np.cos(angles)
        y = cy + ry * np.sin(angles)

        return np.column_stack([x, y])

    def _resample_contour(self, contour: np.ndarray, num_points: int) -> np.ndarray:
        """
        Resample contour to have specified number of points.

        Args:
            contour (np.ndarray): Input contour
            num_points (int): Desired number of points

        Returns:
            np.ndarray: Resampled contour
        """
        if len(contour.shape) == 3:
            contour = contour.squeeze()

        # Compute cumulative arc length
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        distances = np.concatenate([[0], distances])
        arc_length = np.cumsum(distances)

        # Interpolate
        total_length = arc_length[-1]
        target_arc_lengths = np.linspace(0, total_length, num_points, endpoint=False)

        x_interp = np.interp(target_arc_lengths, arc_length, contour[:, 0])
        y_interp = np.interp(target_arc_lengths, arc_length, contour[:, 1])

        return np.column_stack([x_interp, y_interp])

    def contour_to_mask(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert contour to binary mask.

        Args:
            contour (np.ndarray): Contour points (Nx2)
            shape (Tuple[int, int]): Output shape (h, w)

        Returns:
            np.ndarray: Binary mask
        """
        mask = np.zeros(shape, dtype=np.uint8)

        # Convert contour to integer coordinates
        contour_int = np.round(contour).astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [contour_int], 255)

        return mask

    def refine_with_gradient_flow(self, frame: np.ndarray, initial_contour: np.ndarray, iterations: int = 100) -> np.ndarray:
        """
        Refine contour using gradient vector flow.

        Args:
            frame (np.ndarray): Input frame
            initial_contour (np.ndarray): Initial contour
            iterations (int): Number of iterations

        Returns:
            np.ndarray: Refined contour
        """
        # Compute gradient
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)

        # Normalize gradients
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        grad_x_norm = grad_x / grad_mag
        grad_y_norm = grad_y / grad_mag

        # Evolve contour
        contour = initial_contour.copy()

        for _ in range(iterations):
            # For each point, move along gradient
            for i in range(len(contour)):
                x, y = int(contour[i, 0]), int(contour[i, 1])

                # Check bounds
                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                    # Move in gradient direction
                    dx = grad_x_norm[y, x]
                    dy = grad_y_norm[y, x]

                    contour[i, 0] += dx * self.config['gamma']
                    contour[i, 1] += dy * self.config['gamma']

            # Smooth contour
            contour = self._smooth_contour(contour)

        return contour

    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Smooth contour using moving average.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            np.ndarray: Smoothed contour
        """
        window_size = 5
        smoothed = np.copy(contour)

        for i in range(len(contour)):
            # Get neighboring points (circular)
            indices = [(i + j - window_size // 2) % len(contour) for j in range(window_size)]
            neighbors = contour[indices]

            # Average
            smoothed[i] = np.mean(neighbors, axis=0)

        return smoothed

    def detect_with_edge_initialization(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Detect boundary using edge map for initialization.

        Args:
            frame (np.ndarray): Input frame
            edges (np.ndarray): Edge map

        Returns:
            np.ndarray: Detected contour
        """
        # Find contours in edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # No edges found, use default initialization
            return self.detect(frame)

        # Use largest contour for initialization
        largest = max(contours, key=cv2.contourArea)
        initial_contour = self._resample_contour(largest.squeeze(), self.config['init_num_points'])

        # Run snake
        return self.detect(frame, initial_contour)


def create_active_contour_detector(config: Optional[Dict] = None) -> ActiveContourDetector:
    """
    Factory function to create active contour detector.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        ActiveContourDetector: Configured detector
    """
    return ActiveContourDetector(config)


if __name__ == "__main__":
    import sys
    sys.path.append('..')

    if len(sys.argv) > 1:
        from video_loader import VideoLoader
        from preprocessing import UltrasoundPreprocessor
        from roi_detector import ROIDetector

        video_path = sys.argv[1]

        with VideoLoader(video_path) as loader:
            frame = loader.extract_frame(0)

            if frame is not None:
                # Preprocess
                preprocessor = UltrasoundPreprocessor()
                preprocessed = preprocessor.preprocess(frame)

                # Detect ROI
                roi_detector = ROIDetector()
                roi, bbox = roi_detector.detect_roi(preprocessed)

                # Active contour
                detector = ActiveContourDetector()
                contour = detector.detect(roi)

                print(f"Detected contour: {len(contour)} points")
                print(f"Contour shape: {contour.shape}")
    else:
        print("Usage: python active_contour.py <video_path>")
