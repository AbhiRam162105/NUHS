"""
Preprocessing Module

Handles noise reduction, contrast enhancement, and preprocessing for ultrasound images.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from skimage import restoration, filters
from skimage.morphology import disk, square
import logging

logger = logging.getLogger(__name__)


class UltrasoundPreprocessor:
    """
    Preprocesses ultrasound images to reduce noise and enhance boundaries.

    Applies a series of filters and transformations optimized for ultrasound imaging.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        # Default parameters
        self.config = {
            # Speckle reduction
            'median_kernel_size': 5,
            'gaussian_sigma': 1.2,
            'denoise_weight': 0.1,
            'denoise_enabled': False,  # Anisotropic diffusion (slower)

            # Contrast enhancement
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),

            # Morphological
            'tophat_kernel_size': 15,
            'blackhat_kernel_size': 15,
            'morphological_enabled': True,

            # General
            'normalize_output': True
        }

        if config:
            self.config.update(config)

        logger.info(f"Preprocessor initialized with config: {self.config}")

    def preprocess(self, frame: np.ndarray, return_intermediate: bool = False) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a frame.

        Args:
            frame (np.ndarray): Input grayscale frame
            return_intermediate (bool): If True, return dict with intermediate results

        Returns:
            np.ndarray or Dict: Preprocessed frame or dict with intermediate steps
        """
        intermediate = {}
        intermediate['original'] = frame.copy()

        # Step 1: Speckle reduction
        denoised = self._reduce_speckle(frame)
        intermediate['denoised'] = denoised

        # Step 2: Contrast enhancement
        enhanced = self._enhance_contrast(denoised)
        intermediate['enhanced'] = enhanced

        # Step 3: Morphological preprocessing (optional)
        if self.config['morphological_enabled']:
            morphological = self._morphological_preprocessing(enhanced)
            intermediate['morphological'] = morphological
            final = morphological
        else:
            final = enhanced

        # Step 4: Normalize if needed
        if self.config['normalize_output']:
            final = self._normalize(final)

        intermediate['final'] = final

        if return_intermediate:
            return intermediate
        else:
            return final

    def _reduce_speckle(self, frame: np.ndarray) -> np.ndarray:
        """
        Reduce speckle noise using multiple filters.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Denoised frame
        """
        # Median filter - removes salt-and-pepper noise
        kernel_size = self.config['median_kernel_size']
        if kernel_size % 2 == 0:  # Ensure odd
            kernel_size += 1

        denoised = cv2.medianBlur(frame, kernel_size)

        # Gaussian blur - smooths while preserving edges
        sigma = self.config['gaussian_sigma']
        denoised = cv2.GaussianBlur(denoised, (0, 0), sigma)

        # Anisotropic diffusion (optional, more expensive)
        if self.config['denoise_enabled']:
            try:
                # Convert to float for denoise_tv_chambolle
                denoised_float = denoised.astype(np.float32) / 255.0
                denoised_float = restoration.denoise_tv_chambolle(
                    denoised_float,
                    weight=self.config['denoise_weight']
                )
                denoised = (denoised_float * 255).astype(np.uint8)
            except Exception as e:
                logger.warning(f"Anisotropic diffusion failed: {e}")

        return denoised

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Contrast-enhanced frame
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_tile_size']
        )
        enhanced = clahe.apply(frame)
        return enhanced

    def _morphological_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to enhance vessel structures.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Morphologically processed frame
        """
        # Top-hat transform - enhance bright vessels
        tophat_size = self.config['tophat_kernel_size']
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_size, tophat_size))
        tophat = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, tophat_kernel)

        # Black-hat transform - enhance dark vessels
        blackhat_size = self.config['blackhat_kernel_size']
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_size, blackhat_size))
        blackhat = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, blackhat_kernel)

        # Combine: original + tophat - blackhat
        result = cv2.add(frame, tophat)
        result = cv2.subtract(result, blackhat)

        return result

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to full intensity range.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Normalized frame
        """
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    def adaptive_median_filter(self, frame: np.ndarray, max_kernel_size: int = 7) -> np.ndarray:
        """
        Apply adaptive median filter that varies kernel size based on local statistics.

        Args:
            frame (np.ndarray): Input frame
            max_kernel_size (int): Maximum kernel size

        Returns:
            np.ndarray: Filtered frame
        """
        # Simple implementation: use median with adaptive kernel
        result = frame.copy()

        for kernel_size in range(3, max_kernel_size + 1, 2):
            filtered = cv2.medianBlur(frame, kernel_size)
            # Use filtered value where noise is detected
            noise_mask = np.abs(frame.astype(float) - filtered.astype(float)) > 30
            result[noise_mask] = filtered[noise_mask]

        return result

    def bilateral_filter(self, frame: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.

        Args:
            frame (np.ndarray): Input frame
            d (int): Diameter of pixel neighborhood
            sigma_color (float): Filter sigma in color space
            sigma_space (float): Filter sigma in coordinate space

        Returns:
            np.ndarray: Filtered frame
        """
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

    def unsharp_mask(self, frame: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges.

        Args:
            frame (np.ndarray): Input frame
            sigma (float): Gaussian blur sigma
            strength (float): Enhancement strength

        Returns:
            np.ndarray: Sharpened frame
        """
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    def gamma_correction(self, frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            frame (np.ndarray): Input frame
            gamma (float): Gamma value (< 1 brightens, > 1 darkens)

        Returns:
            np.ndarray: Gamma-corrected frame
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)

    def multiscale_preprocessing(self, frame: np.ndarray, scales: list = [0.5, 1.0, 2.0]) -> np.ndarray:
        """
        Apply preprocessing at multiple scales and combine.

        Args:
            frame (np.ndarray): Input frame
            scales (list): List of scale factors

        Returns:
            np.ndarray: Multi-scale processed frame
        """
        h, w = frame.shape
        processed_scales = []

        for scale in scales:
            # Resize
            scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Process
            processed = self.preprocess(scaled)

            # Resize back
            processed_resized = cv2.resize(processed, (w, h), interpolation=cv2.INTER_LINEAR)
            processed_scales.append(processed_resized)

        # Average across scales
        result = np.mean(processed_scales, axis=0).astype(np.uint8)
        return result


def create_preprocessor(config: Optional[Dict] = None) -> UltrasoundPreprocessor:
    """
    Factory function to create a preprocessor.

    Args:
        config (Optional[Dict]): Configuration parameters

    Returns:
        UltrasoundPreprocessor: Configured preprocessor
    """
    return UltrasoundPreprocessor(config)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        from video_loader import VideoLoader

        video_path = sys.argv[1]

        with VideoLoader(video_path) as loader:
            # Extract first frame
            frame = loader.extract_frame(0)

            if frame is not None:
                # Preprocess with intermediate results
                preprocessor = UltrasoundPreprocessor()
                results = preprocessor.preprocess(frame, return_intermediate=True)

                # Display results
                for stage, image in results.items():
                    print(f"{stage}: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
    else:
        print("Usage: python preprocessing.py <video_path>")
