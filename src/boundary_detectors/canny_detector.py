"""
Canny Edge Detector

Edge-based boundary detection using Canny and Sobel algorithms.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CannyBoundaryDetector:
    """
    Detects boundaries using edge-based methods (Canny and Sobel).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Canny parameters
            'canny_low_threshold_ratio': 0.5,  # Ratio to median intensity
            'canny_high_threshold_ratio': 1.5,
            'canny_aperture': 3,  # Sobel kernel size for Canny
            'canny_l2_gradient': True,

            # Sobel parameters
            'sobel_kernel_size': 3,  # 3 or 5
            'sobel_threshold_percentile': 75,  # Threshold at 75th percentile

            # Post-processing
            'dilate_edges': True,
            'dilation_kernel_size': 3,
            'use_nms': True,  # Non-maximum suppression

            # Confidence scoring
            'edge_strength_weight': 0.6,
            'edge_continuity_weight': 0.4
        }

        if config:
            self.config.update(config)

        logger.info(f"Canny detector initialized with config: {self.config}")

    def detect(self, frame: np.ndarray, return_confidence: bool = False) -> np.ndarray:
        """
        Detect boundaries using Canny method.

        Args:
            frame (np.ndarray): Input grayscale frame
            return_confidence (bool): Return confidence map

        Returns:
            np.ndarray: Binary edge map (and optionally confidence map)
        """
        # Adaptive threshold calculation
        median_intensity = np.median(frame)
        low_threshold = int(median_intensity * self.config['canny_low_threshold_ratio'])
        high_threshold = int(median_intensity * self.config['canny_high_threshold_ratio'])

        # Apply Canny
        edges = cv2.Canny(
            frame,
            low_threshold,
            high_threshold,
            apertureSize=self.config['canny_aperture'],
            L2gradient=self.config['canny_l2_gradient']
        )

        # Post-process edges
        if self.config['dilate_edges']:
            edges = self._dilate_edges(edges)

        if return_confidence:
            confidence = self._compute_confidence(frame, edges)
            return edges, confidence
        else:
            return edges

    def detect_sobel(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect boundaries using Sobel gradient.

        Args:
            frame (np.ndarray): Input grayscale frame

        Returns:
            np.ndarray: Binary edge map
        """
        ksize = self.config['sobel_kernel_size']

        # Compute gradients
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=ksize)

        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Non-maximum suppression (optional)
        if self.config['use_nms']:
            angle = np.arctan2(grad_y, grad_x)
            magnitude = self._non_maximum_suppression(magnitude, angle)

        # Threshold at percentile
        threshold = np.percentile(magnitude, self.config['sobel_threshold_percentile'])
        edges = (magnitude > threshold).astype(np.uint8) * 255

        # Post-process
        if self.config['dilate_edges']:
            edges = self._dilate_edges(edges)

        return edges

    def detect_combined(self, frame: np.ndarray) -> np.ndarray:
        """
        Combine Canny and Sobel detections.

        Args:
            frame (np.ndarray): Input grayscale frame

        Returns:
            np.ndarray: Combined edge map
        """
        canny_edges = self.detect(frame)
        sobel_edges = self.detect_sobel(frame)

        # Combine using OR operation (edge if detected by either method)
        combined = cv2.bitwise_or(canny_edges, sobel_edges)

        return combined

    def extract_boundary_contours(self, edges: np.ndarray, min_contour_length: int = 50) -> List[np.ndarray]:
        """
        Extract boundary contours from edge map.

        Args:
            edges (np.ndarray): Binary edge map
            min_contour_length (int): Minimum contour length to keep

        Returns:
            List[np.ndarray]: List of contours
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by length
        valid_contours = [c for c in contours if len(c) >= min_contour_length]

        # Sort by length (longest first)
        valid_contours.sort(key=len, reverse=True)

        return valid_contours

    def get_dominant_boundary(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the dominant (longest/largest) boundary from edges.

        Args:
            edges (np.ndarray): Binary edge map

        Returns:
            Optional[np.ndarray]: Dominant contour or None
        """
        contours = self.extract_boundary_contours(edges)

        if len(contours) == 0:
            return None

        # Return longest contour
        return contours[0]

    def fit_ellipse_to_boundary(self, contour: np.ndarray) -> Optional[Tuple]:
        """
        Fit an ellipse to a boundary contour.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            Optional[Tuple]: Ellipse parameters ((cx, cy), (w, h), angle) or None
        """
        if len(contour) < 5:  # Need at least 5 points for ellipse fitting
            return None

        try:
            ellipse = cv2.fitEllipse(contour)
            return ellipse
        except cv2.error:
            return None

    def _dilate_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Dilate edges to make them more continuous.

        Args:
            edges (np.ndarray): Input edges

        Returns:
            np.ndarray: Dilated edges
        """
        kernel_size = self.config['dilation_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return dilated

    def _non_maximum_suppression(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """
        Apply non-maximum suppression to gradient magnitude.

        Args:
            magnitude (np.ndarray): Gradient magnitude
            angle (np.ndarray): Gradient angle

        Returns:
            np.ndarray: Suppressed magnitude
        """
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convert angle to degrees and normalize to [0, 180)
        angle_deg = np.rad2deg(angle) % 180

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q = 255
                r = 255

                # Angle 0 (horizontal)
                if (0 <= angle_deg[i, j] < 22.5) or (157.5 <= angle_deg[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle_deg[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90 (vertical)
                elif 67.5 <= angle_deg[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle_deg[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    def _compute_confidence(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Compute confidence map for detected edges.

        Args:
            frame (np.ndarray): Input frame
            edges (np.ndarray): Detected edges

        Returns:
            np.ndarray: Confidence map (0-1)
        """
        # Compute gradient magnitude for edge strength
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude_norm = magnitude / (magnitude.max() + 1e-6)

        # Compute edge continuity (local edge density)
        continuity = cv2.GaussianBlur(edges.astype(float) / 255.0, (5, 5), 1.0)

        # Combine
        edge_strength_weight = self.config['edge_strength_weight']
        continuity_weight = self.config['edge_continuity_weight']

        confidence = (edge_strength_weight * magnitude_norm +
                     continuity_weight * continuity)
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def detect_with_hysteresis(self, frame: np.ndarray, low_ratio: float = 0.3, high_ratio: float = 0.7) -> np.ndarray:
        """
        Multi-level edge detection with hysteresis thresholding.

        Args:
            frame (np.ndarray): Input frame
            low_ratio (float): Low threshold ratio
            high_ratio (float): High threshold ratio

        Returns:
            np.ndarray: Edge map with strong and weak edges
        """
        median_intensity = np.median(frame)
        low_threshold = int(median_intensity * low_ratio)
        high_threshold = int(median_intensity * high_ratio)

        # Detect edges at different thresholds
        strong_edges = cv2.Canny(frame, high_threshold, high_threshold * 1.5,
                                 apertureSize=self.config['canny_aperture'])
        weak_edges = cv2.Canny(frame, low_threshold, high_threshold,
                               apertureSize=self.config['canny_aperture'])

        # Combine: strong edges + weak edges connected to strong edges
        combined = cv2.dilate(strong_edges, np.ones((3, 3), np.uint8))
        result = cv2.bitwise_and(weak_edges, combined)
        result = cv2.bitwise_or(result, strong_edges)

        return result


def create_canny_detector(config: Optional[Dict] = None) -> CannyBoundaryDetector:
    """
    Factory function to create Canny detector.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        CannyBoundaryDetector: Configured detector
    """
    return CannyBoundaryDetector(config)


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

                # Detect edges
                detector = CannyBoundaryDetector()
                edges = detector.detect_combined(roi)

                print(f"Detected edges in ROI: shape={edges.shape}, num_edges={np.sum(edges > 0)}")

                # Extract contours
                contours = detector.extract_boundary_contours(edges)
                print(f"Found {len(contours)} contours")
    else:
        print("Usage: python canny_detector.py <video_path>")
