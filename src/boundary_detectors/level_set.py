"""
Level Set Method Detector

Boundary detection using level set evolution methods.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from scipy import ndimage as ndi
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour
from skimage.segmentation import checkerboard_level_set, disk_level_set
import logging

logger = logging.getLogger(__name__)


class LevelSetDetector:
    """
    Detects boundaries using level set methods.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Method selection
            'method': 'chan_vese',  # 'chan_vese', 'geodesic', 'custom'

            # Chan-Vese parameters
            'chan_vese_iterations': 200,
            'chan_vese_smoothing': 3,
            'chan_vese_lambda1': 1.0,
            'chan_vese_lambda2': 1.0,

            # Geodesic active contour parameters
            'gac_iterations': 100,
            'gac_smoothing': 1,
            'gac_threshold': 'auto',  # 'auto' or float value
            'gac_balloon': 1,  # Balloon force (-1 to 1)

            # Initialization
            'init_method': 'checkerboard',  # 'checkerboard', 'disk', 'custom'
            'init_disk_radius_ratio': 0.4,

            # Custom level set parameters
            'dt': 0.5,  # Time step
            'mu': 0.2,  # Coefficient of level set regularization
            'nu': 0.0,  # Coefficient of area term
            'lambda': 5.0,  # Coefficient of edge-based term
        }

        if config:
            self.config.update(config)

        logger.info(f"Level Set detector initialized with config: {self.config}")

    def detect(self, frame: np.ndarray, initial_level_set: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect boundary using level set method.

        Args:
            frame (np.ndarray): Input grayscale frame
            initial_level_set (Optional[np.ndarray]): Initial level set function

        Returns:
            np.ndarray: Binary segmentation mask
        """
        method = self.config['method']

        # Initialize level set if not provided
        if initial_level_set is None:
            initial_level_set = self._initialize_level_set(frame.shape)

        # Normalize frame
        frame_norm = self._normalize_frame(frame)

        # Apply selected method
        if method == 'chan_vese':
            return self._chan_vese(frame_norm, initial_level_set)
        elif method == 'geodesic':
            return self._geodesic_active_contour(frame_norm, initial_level_set)
        elif method == 'custom':
            return self._custom_level_set(frame_norm, initial_level_set)
        else:
            logger.warning(f"Unknown method {method}, using Chan-Vese")
            return self._chan_vese(frame_norm, initial_level_set)

    def _chan_vese(self, frame: np.ndarray, init_level_set: np.ndarray) -> np.ndarray:
        """
        Chan-Vese level set segmentation.

        Args:
            frame (np.ndarray): Normalized frame
            init_level_set (np.ndarray): Initial level set

        Returns:
            np.ndarray: Binary segmentation
        """
        segmentation = morphological_chan_vese(
            frame,
            iterations=self.config['chan_vese_iterations'],
            init_level_set=init_level_set,
            smoothing=self.config['chan_vese_smoothing'],
            lambda1=self.config['chan_vese_lambda1'],
            lambda2=self.config['chan_vese_lambda2']
        )

        return (segmentation > 0).astype(np.uint8) * 255

    def _geodesic_active_contour(self, frame: np.ndarray, init_level_set: np.ndarray) -> np.ndarray:
        """
        Geodesic active contour level set.

        Args:
            frame (np.ndarray): Normalized frame
            init_level_set (np.ndarray): Initial level set

        Returns:
            np.ndarray: Binary segmentation
        """
        # Compute edge indicator function
        gradient = self._compute_gradient_magnitude(frame)

        # Invert gradient (edges should have low values)
        edge_indicator = self._edge_indicator(gradient)

        # Apply geodesic active contour
        segmentation = morphological_geodesic_active_contour(
            edge_indicator,
            iterations=self.config['gac_iterations'],
            init_level_set=init_level_set,
            smoothing=self.config['gac_smoothing'],
            balloon=self.config['gac_balloon'],
            threshold=self.config['gac_threshold']
        )

        return (segmentation > 0).astype(np.uint8) * 255

    def _custom_level_set(self, frame: np.ndarray, init_level_set: np.ndarray) -> np.ndarray:
        """
        Custom level set evolution.

        Args:
            frame (np.ndarray): Normalized frame
            init_level_set (np.ndarray): Initial level set

        Returns:
            np.ndarray: Binary segmentation
        """
        # Convert initial level set to signed distance function
        phi = self._initialize_sdf(init_level_set)

        # Compute edge indicator
        gradient = self._compute_gradient_magnitude(frame)
        edge_indicator = self._edge_indicator(gradient)

        # Evolve level set
        for _ in range(self.config['chan_vese_iterations']):
            phi = self._evolve_level_set(phi, edge_indicator)

            # Re-initialize to signed distance function every 10 iterations
            if _ % 10 == 0:
                phi = self._reinitialize_sdf(phi)

        # Extract zero level set
        segmentation = (phi > 0).astype(np.uint8) * 255

        return segmentation

    def _initialize_level_set(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initialize level set function.

        Args:
            shape (Tuple[int, int]): Image shape

        Returns:
            np.ndarray: Initial level set
        """
        method = self.config['init_method']

        if method == 'checkerboard':
            return checkerboard_level_set(shape)
        elif method == 'disk':
            h, w = shape
            center = (h // 2, w // 2)
            radius = min(h, w) * self.config['init_disk_radius_ratio']
            return disk_level_set(shape, center=center, radius=radius)
        else:
            # Default to checkerboard
            return checkerboard_level_set(shape)

    def _initialize_sdf(self, init_level_set: np.ndarray) -> np.ndarray:
        """
        Initialize signed distance function from binary mask.

        Args:
            init_level_set (np.ndarray): Binary initialization

        Returns:
            np.ndarray: Signed distance function
        """
        # Convert to binary if needed
        if init_level_set.max() > 1:
            binary = (init_level_set > 0).astype(np.uint8)
        else:
            binary = init_level_set.astype(np.uint8)

        # Compute distance transforms
        dist_inside = ndi.distance_transform_edt(binary)
        dist_outside = ndi.distance_transform_edt(1 - binary)

        # Signed distance function
        phi = dist_inside - dist_outside

        return phi

    def _reinitialize_sdf(self, phi: np.ndarray) -> np.ndarray:
        """
        Re-initialize level set to signed distance function.

        Args:
            phi (np.ndarray): Current level set

        Returns:
            np.ndarray: Re-initialized signed distance function
        """
        binary = (phi > 0).astype(np.uint8)
        return self._initialize_sdf(binary)

    def _evolve_level_set(self, phi: np.ndarray, edge_indicator: np.ndarray) -> np.ndarray:
        """
        Evolve level set by one time step.

        Args:
            phi (np.ndarray): Current level set
            edge_indicator (np.ndarray): Edge indicator function

        Returns:
            np.ndarray: Updated level set
        """
        # Compute curvature
        kappa = self._compute_curvature(phi)

        # Compute gradient magnitude
        grad_phi = self._compute_gradient_magnitude(phi)

        # Evolution equation terms
        dt = self.config['dt']
        mu = self.config['mu']
        nu = self.config['nu']
        lambda_param = self.config['lambda']

        # Edge-based term
        edge_term = edge_indicator * kappa * grad_phi

        # Area term
        area_term = nu * edge_indicator

        # Regularization term
        regularization = mu * (4 * ndi.laplace(phi) - kappa)

        # Update
        phi_new = phi + dt * (lambda_param * edge_term + area_term + regularization)

        return phi_new

    def _compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Gradient magnitude
        """
        # Convert to float if needed
        if image.dtype == np.uint8:
            image = image.astype(float) / 255.0

        # Compute gradients
        grad_x = ndi.sobel(image, axis=1)
        grad_y = ndi.sobel(image, axis=0)

        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return magnitude

    def _edge_indicator(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute edge indicator function.

        Args:
            gradient (np.ndarray): Gradient magnitude

        Returns:
            np.ndarray: Edge indicator (low at edges)
        """
        # g(|∇I|) = 1 / (1 + |∇I|²)
        indicator = 1.0 / (1.0 + gradient**2)

        return indicator

    def _compute_curvature(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute mean curvature of level set.

        Args:
            phi (np.ndarray): Level set function

        Returns:
            np.ndarray: Curvature
        """
        # First derivatives
        phi_x = ndi.sobel(phi, axis=1)
        phi_y = ndi.sobel(phi, axis=0)

        # Second derivatives
        phi_xx = ndi.sobel(phi_x, axis=1)
        phi_yy = ndi.sobel(phi_y, axis=0)
        phi_xy = ndi.sobel(phi_x, axis=0)

        # Curvature formula
        numerator = phi_xx * phi_y**2 - 2 * phi_xy * phi_x * phi_y + phi_yy * phi_x**2
        denominator = (phi_x**2 + phi_y**2 + 1e-10) ** 1.5

        curvature = numerator / denominator

        return curvature

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1].

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Normalized frame
        """
        frame_float = frame.astype(float)
        frame_norm = (frame_float - frame_float.min()) / (frame_float.max() - frame_float.min() + 1e-10)
        return frame_norm

    def extract_contour(self, segmentation: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract contour from segmentation.

        Args:
            segmentation (np.ndarray): Binary segmentation

        Returns:
            Optional[np.ndarray]: Contour or None
        """
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Return largest contour
        largest = max(contours, key=cv2.contourArea)
        return largest

    def detect_multiscale(self, frame: np.ndarray, scales: list = [0.5, 1.0]) -> np.ndarray:
        """
        Multi-scale level set detection.

        Args:
            frame (np.ndarray): Input frame
            scales (list): List of scales

        Returns:
            np.ndarray: Combined segmentation
        """
        h, w = frame.shape
        segmentations = []

        for scale in scales:
            # Resize
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))

            # Detect
            scaled_seg = self.detect(scaled_frame)

            # Resize back
            seg_resized = cv2.resize(scaled_seg, (w, h), interpolation=cv2.INTER_NEAREST)
            segmentations.append(seg_resized)

        # Combine by voting
        combined = np.mean(segmentations, axis=0)
        combined_binary = (combined > 127).astype(np.uint8) * 255

        return combined_binary


def create_level_set_detector(config: Optional[Dict] = None) -> LevelSetDetector:
    """
    Factory function to create level set detector.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        LevelSetDetector: Configured detector
    """
    return LevelSetDetector(config)


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

                # Level set
                detector = LevelSetDetector()
                segmentation = detector.detect(roi)

                print(f"Level set segmentation complete")
                print(f"Segmentation shape: {segmentation.shape}")
                print(f"Vessel pixels: {np.sum(segmentation > 0)}")

                # Extract contour
                contour = detector.extract_contour(segmentation)
                if contour is not None:
                    print(f"Contour points: {len(contour)}")
    else:
        print("Usage: python level_set.py <video_path>")
