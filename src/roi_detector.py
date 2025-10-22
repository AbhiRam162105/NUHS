"""
ROI Detector Module

Detects and extracts Region of Interest (ROI) containing vessels in ultrasound frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class ROIDetector:
    """
    Detects Region of Interest in ultrasound images.

    Uses thresholding and morphological operations to locate vessel regions.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ROI detector.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Thresholding
            'threshold_method': 'otsu',  # 'otsu', 'adaptive', 'multi_otsu'
            'invert_threshold': True,  # True for dark vessels (anechoic)

            # Component filtering
            'min_area': 500,  # Minimum area in pixels
            'max_area': 100000,  # Maximum area in pixels
            'min_circularity': 0.3,  # 0-1, higher = more circular
            'max_circularity': 1.0,
            'min_aspect_ratio': 0.3,  # width/height
            'max_aspect_ratio': 3.0,

            # ROI expansion
            'padding_percent': 0.25,  # Add 25% padding around detected region

            # Morphological operations
            'morph_open_kernel': 5,
            'morph_close_kernel': 11,

            # Fallback
            'use_center_fallback': True,  # If no ROI found, use center region
            'fallback_size_percent': 0.6  # 60% of image centered
        }

        if config:
            self.config.update(config)

        logger.info(f"ROI Detector initialized with config: {self.config}")

    def detect_roi(self, frame: np.ndarray, return_debug_info: bool = False) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect ROI in the frame.

        Args:
            frame (np.ndarray): Input grayscale frame
            return_debug_info (bool): Return debug information

        Returns:
            Tuple containing:
                - np.ndarray: Cropped ROI
                - Tuple[int, int, int, int]: Bounding box (x, y, w, h)
                - (optional) Dict: Debug information
        """
        h, w = frame.shape

        # Step 1: Threshold to find vessel candidates
        binary = self._threshold_frame(frame)

        # Step 2: Morphological operations to clean up
        cleaned = self._morphological_cleanup(binary)

        # Step 3: Find connected components
        components = self._find_components(cleaned)

        # Step 4: Filter components by criteria
        valid_components = self._filter_components(components, frame.shape)

        # Step 5: Select best ROI
        bbox = self._select_best_roi(valid_components, frame.shape)

        # Step 6: Expand ROI with padding
        bbox_expanded = self._expand_bbox(bbox, frame.shape)

        # Extract ROI
        x, y, w, h = bbox_expanded
        roi = frame[y:y+h, x:x+w]

        if return_debug_info:
            debug_info = {
                'binary': binary,
                'cleaned': cleaned,
                'num_components': len(components),
                'valid_components': len(valid_components),
                'bbox_original': bbox,
                'bbox_expanded': bbox_expanded
            }
            return roi, bbox_expanded, debug_info
        else:
            return roi, bbox_expanded

    def _threshold_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to segment frame.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Binary mask
        """
        method = self.config['threshold_method']

        if method == 'otsu':
            # Otsu's method
            _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'multi_otsu':
            # Multi-level Otsu (using scikit-image)
            try:
                from skimage.filters import threshold_multiotsu
                thresholds = threshold_multiotsu(frame, classes=3)
                binary = np.digitize(frame, bins=thresholds)
                binary = (binary == 0).astype(np.uint8) * 255  # Take darkest region
            except ImportError:
                logger.warning("scikit-image not available, falling back to Otsu")
                _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Default to Otsu
            _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if looking for dark vessels
        if self.config['invert_threshold']:
            binary = cv2.bitwise_not(binary)

        return binary

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean binary mask.

        Args:
            binary (np.ndarray): Binary mask

        Returns:
            np.ndarray: Cleaned binary mask
        """
        # Opening: removes small noise
        open_size = self.config['morph_open_kernel']
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        # Closing: fills small gaps
        close_size = self.config['morph_close_kernel']
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

        return closed

    def _find_components(self, binary: np.ndarray) -> List[Dict]:
        """
        Find connected components in binary mask.

        Args:
            binary (np.ndarray): Binary mask

        Returns:
            List[Dict]: List of component properties
        """
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        components = []
        for contour in contours:
            # Calculate properties
            area = cv2.contourArea(contour)
            if area == 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Circularity = 4π * area / perimeter^2
            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2

            components.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'centroid': (cx, cy)
            })

        return components

    def _filter_components(self, components: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Filter components by criteria.

        Args:
            components (List[Dict]): List of components
            frame_shape (Tuple[int, int]): Frame shape (h, w)

        Returns:
            List[Dict]: Filtered components
        """
        valid = []

        for comp in components:
            # Check area
            if comp['area'] < self.config['min_area'] or comp['area'] > self.config['max_area']:
                continue

            # Check circularity
            if comp['circularity'] < self.config['min_circularity'] or comp['circularity'] > self.config['max_circularity']:
                continue

            # Check aspect ratio
            if comp['aspect_ratio'] < self.config['min_aspect_ratio'] or comp['aspect_ratio'] > self.config['max_aspect_ratio']:
                continue

            valid.append(comp)

        # Sort by area (largest first)
        valid.sort(key=lambda x: x['area'], reverse=True)

        return valid

    def _select_best_roi(self, components: List[Dict], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Select the best ROI from valid components.

        Args:
            components (List[Dict]): Valid components
            frame_shape (Tuple[int, int]): Frame shape (h, w)

        Returns:
            Tuple[int, int, int, int]: Bounding box (x, y, w, h)
        """
        if len(components) == 0:
            # No valid components - use fallback
            if self.config['use_center_fallback']:
                return self._get_center_roi(frame_shape)
            else:
                # Return full frame
                h, w = frame_shape
                return (0, 0, w, h)

        # Select largest component (first after sorting)
        best = components[0]
        return best['bbox']

    def _get_center_roi(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get centered ROI as fallback.

        Args:
            frame_shape (Tuple[int, int]): Frame shape (h, w)

        Returns:
            Tuple[int, int, int, int]: Bounding box (x, y, w, h)
        """
        h, w = frame_shape
        size_percent = self.config['fallback_size_percent']

        roi_w = int(w * size_percent)
        roi_h = int(h * size_percent)
        x = (w - roi_w) // 2
        y = (h - roi_h) // 2

        logger.info(f"Using center fallback ROI: ({x}, {y}, {roi_w}, {roi_h})")
        return (x, y, roi_w, roi_h)

    def _expand_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Expand bounding box with padding.

        Args:
            bbox (Tuple[int, int, int, int]): Original bbox (x, y, w, h)
            frame_shape (Tuple[int, int]): Frame shape (h, w)

        Returns:
            Tuple[int, int, int, int]: Expanded bbox
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape

        padding = self.config['padding_percent']

        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        # Expand
        x_new = max(0, x - pad_w)
        y_new = max(0, y - pad_h)
        w_new = min(frame_w - x_new, w + 2 * pad_w)
        h_new = min(frame_h - y_new, h + 2 * pad_h)

        return (x_new, y_new, w_new, h_new)

    def detect_multiple_rois(self, frame: np.ndarray, max_rois: int = 3) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect multiple ROIs in the frame.

        Args:
            frame (np.ndarray): Input frame
            max_rois (int): Maximum number of ROIs to return

        Returns:
            List[Tuple[np.ndarray, Tuple]]: List of (roi, bbox) tuples
        """
        # Threshold and clean
        binary = self._threshold_frame(frame)
        cleaned = self._morphological_cleanup(binary)

        # Find components
        components = self._find_components(cleaned)
        valid_components = self._filter_components(components, frame.shape)

        # Take top N components
        rois = []
        for comp in valid_components[:max_rois]:
            bbox = comp['bbox']
            bbox_expanded = self._expand_bbox(bbox, frame.shape)
            x, y, w, h = bbox_expanded
            roi = frame[y:y+h, x:x+w]
            rois.append((roi, bbox_expanded))

        return rois

    def visualize_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Visualize ROI on frame.

        Args:
            frame (np.ndarray): Input frame
            bbox (Tuple[int, int, int, int]): Bounding box

        Returns:
            np.ndarray: Frame with ROI drawn
        """
        # Convert to BGR for colored rectangle
        if len(frame.shape) == 2:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = frame.copy()

        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return vis


def create_roi_detector(config: Optional[Dict] = None) -> ROIDetector:
    """
    Factory function to create ROI detector.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        ROIDetector: Configured detector
    """
    return ROIDetector(config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from video_loader import VideoLoader
        from preprocessing import UltrasoundPreprocessor

        video_path = sys.argv[1]

        with VideoLoader(video_path) as loader:
            frame = loader.extract_frame(0)

            if frame is not None:
                # Preprocess
                preprocessor = UltrasoundPreprocessor()
                preprocessed = preprocessor.preprocess(frame)

                # Detect ROI
                detector = ROIDetector()
                roi, bbox, debug_info = detector.detect_roi(preprocessed, return_debug_info=True)

                print(f"ROI detected: bbox={bbox}, roi_shape={roi.shape}")
                print(f"Debug info: {debug_info}")
    else:
        print("Usage: python roi_detector.py <video_path>")
