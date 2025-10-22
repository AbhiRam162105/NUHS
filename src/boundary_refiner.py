"""
Boundary Refiner Module

Combines and refines boundary detections from multiple methods.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial import distance
import logging

logger = logging.getLogger(__name__)


class BoundaryRefiner:
    """
    Refines and fuses boundaries detected by multiple methods.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the refiner.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Fusion method
            'fusion_method': 'voting',  # 'voting', 'weighted_average', 'confidence_based'
            'voting_threshold': 2,  # Number of methods that must agree

            # Weights for different detectors (if using weighted fusion)
            'detector_weights': {
                'canny': 0.2,
                'active_contour': 0.3,
                'watershed': 0.25,
                'level_set': 0.25
            },

            # Morphological refinement
            'apply_morphological': True,
            'morph_close_kernel': 5,
            'morph_open_kernel': 3,

            # Contour approximation
            'approximate_contour': True,
            'approx_epsilon_ratio': 0.005,  # Ratio of contour perimeter

            # Shape fitting
            'fit_shape': 'none',  # 'none', 'ellipse', 'circle'

            # Quality filtering
            'min_circularity': 0.4,
            'max_circularity': 1.0,
            'min_area': 100,
            'max_area': 1000000,

            # Boundary smoothing
            'smooth_boundary': True,
            'smoothing_window': 5
        }

        if config:
            self.config.update(config)

        logger.info(f"Boundary Refiner initialized with config: {self.config}")

    def fuse_boundaries(self, boundaries: Dict[str, np.ndarray],
                       confidences: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Fuse multiple boundary detections.

        Args:
            boundaries (Dict[str, np.ndarray]): Dictionary of detector name -> binary boundary map
            confidences (Optional[Dict[str, float]]): Confidence scores for each detector

        Returns:
            np.ndarray: Fused binary boundary map
        """
        method = self.config['fusion_method']

        if method == 'voting':
            return self._voting_fusion(boundaries)
        elif method == 'weighted_average':
            weights = self.config['detector_weights']
            return self._weighted_fusion(boundaries, weights)
        elif method == 'confidence_based' and confidences is not None:
            return self._confidence_fusion(boundaries, confidences)
        else:
            logger.warning(f"Unknown fusion method {method}, using voting")
            return self._voting_fusion(boundaries)

    def _voting_fusion(self, boundaries: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Voting-based fusion: boundary present if K methods agree.

        Args:
            boundaries (Dict[str, np.ndarray]): Boundary maps

        Returns:
            np.ndarray: Fused boundary
        """
        # Stack all boundaries
        boundary_stack = np.stack([b.astype(float) / 255.0 for b in boundaries.values()], axis=0)

        # Count votes
        votes = np.sum(boundary_stack, axis=0)

        # Threshold
        threshold = self.config['voting_threshold']
        fused = (votes >= threshold).astype(np.uint8) * 255

        return fused

    def _weighted_fusion(self, boundaries: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Weighted average fusion.

        Args:
            boundaries (Dict[str, np.ndarray]): Boundary maps
            weights (Dict[str, float]): Weights for each detector

        Returns:
            np.ndarray: Fused boundary
        """
        weighted_sum = np.zeros_like(next(iter(boundaries.values())), dtype=float)
        total_weight = 0.0

        for name, boundary in boundaries.items():
            weight = weights.get(name, 1.0)
            weighted_sum += boundary.astype(float) / 255.0 * weight
            total_weight += weight

        # Average and threshold
        averaged = weighted_sum / total_weight
        fused = (averaged > 0.5).astype(np.uint8) * 255

        return fused

    def _confidence_fusion(self, boundaries: Dict[str, np.ndarray],
                          confidences: Dict[str, float]) -> np.ndarray:
        """
        Confidence-based fusion.

        Args:
            boundaries (Dict[str, np.ndarray]): Boundary maps
            confidences (Dict[str, float]): Confidence scores

        Returns:
            np.ndarray: Fused boundary
        """
        weighted_sum = np.zeros_like(next(iter(boundaries.values())), dtype=float)
        total_confidence = sum(confidences.values())

        for name, boundary in boundaries.items():
            confidence = confidences.get(name, 0.5)
            weighted_sum += boundary.astype(float) / 255.0 * confidence

        # Average and threshold
        averaged = weighted_sum / total_confidence
        fused = (averaged > 0.5).astype(np.uint8) * 255

        return fused

    def refine(self, boundary: np.ndarray) -> np.ndarray:
        """
        Apply full refinement pipeline to a boundary.

        Args:
            boundary (np.ndarray): Binary boundary map

        Returns:
            np.ndarray: Refined boundary
        """
        refined = boundary.copy()

        # Step 1: Morphological refinement
        if self.config['apply_morphological']:
            refined = self._morphological_refinement(refined)

        # Step 2: Extract and filter contours
        contours = self._extract_contours(refined)
        if len(contours) == 0:
            logger.warning("No contours found after refinement")
            return refined

        # Step 3: Filter by quality
        valid_contours = self._filter_contours(contours)
        if len(valid_contours) == 0:
            logger.warning("No valid contours after quality filtering")
            return refined

        # Step 4: Select best contour
        best_contour = self._select_best_contour(valid_contours)

        # Step 5: Smooth contour
        if self.config['smooth_boundary']:
            best_contour = self._smooth_contour(best_contour)

        # Step 6: Approximate contour
        if self.config['approximate_contour']:
            best_contour = self._approximate_contour(best_contour)

        # Step 7: Fit shape if requested
        if self.config['fit_shape'] != 'none':
            best_contour = self._fit_shape(best_contour, refined.shape)

        # Convert back to binary map
        result = np.zeros_like(refined)
        cv2.drawContours(result, [best_contour], 0, 255, -1)

        return result

    def _morphological_refinement(self, boundary: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up boundary.

        Args:
            boundary (np.ndarray): Binary boundary

        Returns:
            np.ndarray: Cleaned boundary
        """
        # Closing: fill small gaps
        close_size = self.config['morph_close_kernel']
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, close_kernel)

        # Opening: remove small noise
        open_size = self.config['morph_open_kernel']
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

        return opened

    def _extract_contours(self, boundary: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from boundary.

        Args:
            boundary (np.ndarray): Binary boundary

        Returns:
            List[np.ndarray]: List of contours
        """
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours by quality criteria.

        Args:
            contours (List[np.ndarray]): Input contours

        Returns:
            List[np.ndarray]: Filtered contours
        """
        valid = []

        for contour in contours:
            # Check area
            area = cv2.contourArea(contour)
            if area < self.config['min_area'] or area > self.config['max_area']:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.config['min_circularity'] or circularity > self.config['max_circularity']:
                continue

            valid.append(contour)

        return valid

    def _select_best_contour(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        Select the best contour from a list.

        Args:
            contours (List[np.ndarray]): Input contours

        Returns:
            np.ndarray: Best contour
        """
        # Select largest by area
        best = max(contours, key=cv2.contourArea)
        return best

    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Smooth contour using moving average.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            np.ndarray: Smoothed contour
        """
        if len(contour.shape) == 3:
            contour = contour.squeeze()

        window = self.config['smoothing_window']
        smoothed = np.copy(contour)

        for i in range(len(contour)):
            # Get neighboring points (circular)
            indices = [(i + j - window // 2) % len(contour) for j in range(window)]
            neighbors = contour[indices]

            # Average
            smoothed[i] = np.mean(neighbors, axis=0)

        return smoothed.reshape(-1, 1, 2).astype(np.int32)

    def _approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Approximate contour using Douglas-Peucker algorithm.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            np.ndarray: Approximated contour
        """
        epsilon = self.config['approx_epsilon_ratio'] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def _fit_shape(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Fit a shape (ellipse or circle) to the contour.

        Args:
            contour (np.ndarray): Input contour
            shape (Tuple[int, int]): Image shape

        Returns:
            np.ndarray: Fitted shape contour
        """
        fit_type = self.config['fit_shape']

        if len(contour) < 5:
            return contour

        if fit_type == 'ellipse':
            try:
                ellipse = cv2.fitEllipse(contour)
                # Create ellipse contour
                mask = np.zeros(shape, dtype=np.uint8)
                cv2.ellipse(mask, ellipse, 255, -1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours[0] if len(contours) > 0 else contour
            except cv2.error:
                return contour

        elif fit_type == 'circle':
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Create circle contour
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours[0] if len(contours) > 0 else contour

        else:
            return contour

    def compute_boundary_metrics(self, contour: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for a boundary contour.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            Dict[str, float]: Metrics dictionary
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        # Fit ellipse if possible
        if len(contour) >= 5:
            try:
                (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            except cv2.error:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0

        # Compute smoothness (average curvature variation)
        if len(contour) > 10:
            smoothness = self._compute_smoothness(contour)
        else:
            smoothness = 0.0

        metrics = {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'smoothness': smoothness
        }

        return metrics

    def _compute_smoothness(self, contour: np.ndarray) -> float:
        """
        Compute smoothness measure based on curvature variation.

        Args:
            contour (np.ndarray): Input contour

        Returns:
            float: Smoothness measure
        """
        if len(contour.shape) == 3:
            contour = contour.squeeze()

        # Compute curvatures at each point
        curvatures = []

        for i in range(len(contour)):
            i_prev = (i - 1) % len(contour)
            i_next = (i + 1) % len(contour)

            p_prev = contour[i_prev]
            p_curr = contour[i]
            p_next = contour[i_next]

            # Compute angle
            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            curvatures.append(abs(angle))

        # Smoothness = standard deviation of curvatures (lower = smoother)
        smoothness = np.std(curvatures)

        return smoothness

    def ensemble_refine(self, boundaries: Dict[str, np.ndarray],
                       frame: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Complete ensemble refinement pipeline.

        Args:
            boundaries (Dict[str, np.ndarray]): Multiple boundary detections
            frame (Optional[np.ndarray]): Original frame (for visualization)

        Returns:
            Tuple containing:
                - Refined boundary
                - Metrics dictionary
        """
        # Fuse boundaries
        fused = self.fuse_boundaries(boundaries)

        # Refine
        refined = self.refine(fused)

        # Extract contour and compute metrics
        contours = self._extract_contours(refined)

        if len(contours) > 0:
            best_contour = self._select_best_contour(contours)
            metrics = self.compute_boundary_metrics(best_contour)
        else:
            metrics = {}

        return refined, metrics


def create_boundary_refiner(config: Optional[Dict] = None) -> BoundaryRefiner:
    """
    Factory function to create boundary refiner.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        BoundaryRefiner: Configured refiner
    """
    return BoundaryRefiner(config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from video_loader import VideoLoader
        from preprocessing import UltrasoundPreprocessor
        from roi_detector import ROIDetector
        from boundary_detectors import (
            create_canny_detector,
            create_watershed_detector
        )

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

                # Detect boundaries with multiple methods
                canny = create_canny_detector()
                watershed = create_watershed_detector()

                boundaries = {
                    'canny': canny.detect_combined(roi),
                    'watershed': watershed.extract_boundaries(watershed.detect(roi))
                }

                # Refine
                refiner = BoundaryRefiner()
                refined, metrics = refiner.ensemble_refine(boundaries, roi)

                print(f"Boundary refined successfully")
                print(f"Metrics: {metrics}")
    else:
        print("Usage: python boundary_refiner.py <video_path>")
