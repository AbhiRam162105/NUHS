"""
Watershed Segmentation Detector

Boundary detection using watershed segmentation algorithm.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import logging

logger = logging.getLogger(__name__)


class WatershedDetector:
    """
    Detects boundaries using watershed segmentation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Thresholding for marker detection
            'threshold_method': 'otsu',  # 'otsu', 'adaptive', 'manual'
            'manual_threshold': 127,
            'invert': True,  # For dark vessels

            # Distance transform
            'distance_transform_type': cv2.DIST_L2,
            'distance_mask_size': 5,

            # Marker detection
            'min_distance_between_peaks': 20,  # Minimum distance between markers
            'peak_footprint_size': 3,  # Footprint for peak detection

            # Morphological preprocessing
            'apply_opening': True,
            'opening_kernel_size': 3,
            'apply_closing': True,
            'closing_kernel_size': 5,

            # Watershed parameters
            'compactness': 0.0,  # Compactness of watershed basins

            # Post-processing
            'extract_vessel_region': True,  # Extract vessel (not background)
            'min_region_area': 100
        }

        if config:
            self.config.update(config)

        logger.info(f"Watershed detector initialized with config: {self.config}")

    def detect(self, frame: np.ndarray, return_markers: bool = False) -> np.ndarray:
        """
        Detect boundaries using watershed segmentation.

        Args:
            frame (np.ndarray): Input grayscale frame
            return_markers (bool): Also return marker labels

        Returns:
            np.ndarray: Segmentation labels (and optionally markers)
        """
        # Step 1: Threshold
        binary = self._threshold(frame)

        # Step 2: Morphological preprocessing
        cleaned = self._morphological_preprocess(binary)

        # Step 3: Distance transform
        dist_transform = self._compute_distance_transform(cleaned)

        # Step 4: Find markers
        markers = self._find_markers(dist_transform, cleaned)

        # Step 5: Apply watershed
        labels = watershed(-dist_transform, markers, mask=cleaned,
                          compactness=self.config['compactness'])

        if return_markers:
            return labels, markers
        else:
            return labels

    def extract_boundaries(self, labels: np.ndarray) -> np.ndarray:
        """
        Extract boundaries between watershed regions.

        Args:
            labels (np.ndarray): Watershed labels

        Returns:
            np.ndarray: Binary boundary map
        """
        # Compute gradient of labels to find boundaries
        grad_x = np.abs(np.diff(labels, axis=1, prepend=0))
        grad_y = np.abs(np.diff(labels, axis=0, prepend=0))

        boundaries = ((grad_x + grad_y) > 0).astype(np.uint8) * 255

        return boundaries

    def extract_vessel_mask(self, frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Extract vessel region from watershed labels.

        Args:
            frame (np.ndarray): Original frame
            labels (np.ndarray): Watershed labels

        Returns:
            np.ndarray: Binary vessel mask
        """
        # Find the darkest region (assuming vessel is dark/anechoic)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

        if len(unique_labels) == 0:
            return np.zeros_like(frame, dtype=np.uint8)

        # Compute mean intensity for each region
        region_intensities = []
        for label in unique_labels:
            mask = (labels == label)
            mean_intensity = np.mean(frame[mask])
            region_intensities.append((label, mean_intensity))

        # Sort by intensity (darkest first)
        if self.config['invert']:
            region_intensities.sort(key=lambda x: x[1])
        else:
            region_intensities.sort(key=lambda x: x[1], reverse=True)

        # Select vessel region (darkest or brightest depending on config)
        vessel_label = region_intensities[0][0]
        vessel_mask = (labels == vessel_label).astype(np.uint8) * 255

        return vessel_mask

    def extract_vessel_contour(self, vessel_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract contour from vessel mask.

        Args:
            vessel_mask (np.ndarray): Binary vessel mask

        Returns:
            Optional[np.ndarray]: Vessel contour or None
        """
        contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Return largest contour
        largest = max(contours, key=cv2.contourArea)
        return largest

    def detect_full_pipeline(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run full watershed pipeline.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            Tuple containing:
                - labels: Watershed segmentation labels
                - vessel_mask: Binary vessel mask
                - contour: Vessel contour (or None)
        """
        # Detect watershed labels
        labels = self.detect(frame)

        # Extract vessel mask
        vessel_mask = self.extract_vessel_mask(frame, labels)

        # Extract contour
        contour = self.extract_vessel_contour(vessel_mask)

        return labels, vessel_mask, contour

    def _threshold(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply thresholding.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Binary mask
        """
        method = self.config['threshold_method']

        if method == 'otsu':
            _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        elif method == 'manual':
            _, binary = cv2.threshold(frame, self.config['manual_threshold'], 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.config['invert']:
            binary = cv2.bitwise_not(binary)

        return binary

    def _morphological_preprocess(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations.

        Args:
            binary (np.ndarray): Binary mask

        Returns:
            np.ndarray: Cleaned mask
        """
        result = binary.copy()

        if self.config['apply_opening']:
            kernel_size = self.config['opening_kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

        if self.config['apply_closing']:
            kernel_size = self.config['closing_kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        return result

    def _compute_distance_transform(self, binary: np.ndarray) -> np.ndarray:
        """
        Compute distance transform.

        Args:
            binary (np.ndarray): Binary mask

        Returns:
            np.ndarray: Distance transform
        """
        dist_type = self.config['distance_transform_type']
        mask_size = self.config['distance_mask_size']

        dist_transform = cv2.distanceTransform(binary, dist_type, mask_size)

        return dist_transform

    def _find_markers(self, dist_transform: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Find markers for watershed using distance transform peaks.

        Args:
            dist_transform (np.ndarray): Distance transform
            mask (np.ndarray): Binary mask

        Returns:
            np.ndarray: Marker labels
        """
        min_distance = self.config['min_distance_between_peaks']
        footprint_size = self.config['peak_footprint_size']

        # Find local maxima
        local_max = peak_local_max(
            dist_transform,
            min_distance=min_distance,
            footprint=np.ones((footprint_size, footprint_size)),
            labels=mask
        )

        # Create markers
        markers = np.zeros_like(dist_transform, dtype=np.int32)

        for i, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = i

        # Dilate markers slightly to ensure they're visible
        markers = ndi.maximum_filter(markers, size=3)

        return markers

    def detect_with_custom_markers(self, frame: np.ndarray, markers: np.ndarray) -> np.ndarray:
        """
        Watershed with custom markers.

        Args:
            frame (np.ndarray): Input frame
            markers (np.ndarray): Custom marker labels

        Returns:
            np.ndarray: Segmentation labels
        """
        # Threshold
        binary = self._threshold(frame)

        # Distance transform
        dist_transform = self._compute_distance_transform(binary)

        # Watershed
        labels = watershed(-dist_transform, markers, mask=binary,
                          compactness=self.config['compactness'])

        return labels

    def detect_marker_controlled(self, frame: np.ndarray, foreground_markers: List[Tuple[int, int]],
                                 background_markers: List[Tuple[int, int]]) -> np.ndarray:
        """
        Marker-controlled watershed with explicit foreground/background markers.

        Args:
            frame (np.ndarray): Input frame
            foreground_markers (List[Tuple[int, int]]): List of (y, x) coordinates for foreground
            background_markers (List[Tuple[int, int]]): List of (y, x) coordinates for background

        Returns:
            np.ndarray: Segmentation labels
        """
        # Create marker image
        markers = np.zeros(frame.shape, dtype=np.int32)

        # Mark foreground (label 1)
        for y, x in foreground_markers:
            markers[y, x] = 1

        # Mark background (label 2)
        for y, x in background_markers:
            markers[y, x] = 2

        # Dilate markers
        markers = ndi.maximum_filter(markers, size=5)

        # Compute gradient magnitude for watershed
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Watershed on gradient
        labels = watershed(gradient, markers)

        return labels

    def visualize_segmentation(self, frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Visualize watershed segmentation.

        Args:
            frame (np.ndarray): Original frame
            labels (np.ndarray): Watershed labels

        Returns:
            np.ndarray: Colored segmentation overlay
        """
        # Create colored label image
        colored = np.zeros((*labels.shape, 3), dtype=np.uint8)

        # Assign random colors to each label
        unique_labels = np.unique(labels)
        colors = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)

        for i, label in enumerate(unique_labels):
            colored[labels == label] = colors[i]

        # Overlay on original frame
        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        overlay = cv2.addWeighted(frame_bgr, 0.7, colored, 0.3, 0)

        return overlay


def create_watershed_detector(config: Optional[Dict] = None) -> WatershedDetector:
    """
    Factory function to create watershed detector.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        WatershedDetector: Configured detector
    """
    return WatershedDetector(config)


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

                # Watershed
                detector = WatershedDetector()
                labels, vessel_mask, contour = detector.detect_full_pipeline(roi)

                print(f"Watershed segmentation complete")
                print(f"Number of regions: {len(np.unique(labels))}")
                print(f"Vessel mask shape: {vessel_mask.shape}")
                if contour is not None:
                    print(f"Contour points: {len(contour)}")
    else:
        print("Usage: python watershed_detector.py <video_path>")
