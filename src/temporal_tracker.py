"""
Temporal Tracker Module

Tracks vessel boundaries across video frames for temporal consistency.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TemporalTracker:
    """
    Tracks boundaries temporally across video frames.

    Uses optical flow and Kalman filtering for smooth tracking.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the tracker.

        Args:
            config (Optional[Dict]): Configuration parameters
        """
        self.config = {
            # Optical flow parameters
            'optical_flow_method': 'lucas_kanade',  # 'lucas_kanade', 'farneback'
            'lk_win_size': (15, 15),
            'lk_max_level': 2,
            'lk_criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),

            # Farneback parameters
            'fb_pyr_scale': 0.5,
            'fb_levels': 3,
            'fb_winsize': 15,
            'fb_iterations': 3,
            'fb_poly_n': 5,
            'fb_poly_sigma': 1.2,

            # Kalman filter parameters
            'use_kalman': True,
            'process_noise': 0.05,
            'measurement_noise': 0.1,

            # Temporal consistency
            'max_displacement_per_frame': 25,  # pixels
            'max_area_change_percent': 0.30,  # 30%

            # Tracking confidence
            'min_confidence': 0.5,
            'confidence_decay': 0.95,

            # History
            'history_length': 10,  # Number of frames to keep in history

            # Re-initialization
            'reinit_on_lost': True,
            'lost_threshold': 0.3
        }

        if config:
            self.config.update(config)

        # State
        self.previous_frame = None
        self.previous_contour = None
        self.previous_bbox = None
        self.previous_area = None
        self.confidence = 1.0

        # History
        self.contour_history = deque(maxlen=self.config['history_length'])
        self.area_history = deque(maxlen=self.config['history_length'])
        self.bbox_history = deque(maxlen=self.config['history_length'])

        # Kalman filter
        if self.config['use_kalman']:
            self.kalman = self._initialize_kalman()
        else:
            self.kalman = None

        logger.info(f"Temporal Tracker initialized with config: {self.config}")

    def track(self, current_frame: np.ndarray, detected_contour: Optional[np.ndarray] = None,
              detected_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Track boundary in current frame.

        Args:
            current_frame (np.ndarray): Current frame
            detected_contour (Optional[np.ndarray]): Detected contour in current frame
            detected_bbox (Optional[Tuple]): Detected bbox in current frame

        Returns:
            Tuple containing:
                - Tracked contour
                - Tracked bbox
                - Tracking confidence
        """
        # First frame - initialize
        if self.previous_frame is None:
            return self._initialize_tracking(current_frame, detected_contour, detected_bbox)

        # Predict using optical flow
        predicted_contour = None
        if self.previous_contour is not None:
            predicted_contour = self._predict_with_optical_flow(current_frame)

        # Predict using Kalman filter
        if self.kalman is not None and self.previous_bbox is not None:
            kalman_bbox = self._predict_with_kalman()
        else:
            kalman_bbox = None

        # Combine predictions with detection
        tracked_contour, tracked_bbox = self._fuse_predictions(
            detected_contour, detected_bbox,
            predicted_contour, kalman_bbox
        )

        # Validate tracking
        is_valid = self._validate_tracking(tracked_contour, tracked_bbox)

        if not is_valid:
            logger.warning("Tracking validation failed")
            self.confidence *= self.config['confidence_decay']

            # Re-initialize if confidence too low
            if self.confidence < self.config['lost_threshold'] and self.config['reinit_on_lost']:
                logger.info("Re-initializing tracker")
                return self._initialize_tracking(current_frame, detected_contour, detected_bbox)
        else:
            # Update confidence
            self.confidence = min(1.0, self.confidence + 0.1)

        # Update state
        self._update_state(current_frame, tracked_contour, tracked_bbox)

        return tracked_contour, tracked_bbox, self.confidence

    def _initialize_tracking(self, frame: np.ndarray, contour: Optional[np.ndarray],
                            bbox: Optional[Tuple]) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Initialize tracking with first frame.

        Args:
            frame (np.ndarray): Frame
            contour (Optional[np.ndarray]): Contour
            bbox (Optional[Tuple]): Bounding box

        Returns:
            Tuple of (contour, bbox, confidence)
        """
        self.previous_frame = frame.copy()
        self.previous_contour = contour.copy() if contour is not None else None
        self.previous_bbox = bbox
        self.confidence = 1.0

        if contour is not None:
            self.previous_area = cv2.contourArea(contour)
            self.area_history.append(self.previous_area)

        if contour is not None:
            self.contour_history.append(contour.copy())

        if bbox is not None:
            self.bbox_history.append(bbox)

            # Initialize Kalman filter
            if self.kalman is not None:
                self._init_kalman_state(bbox)

        return self.previous_contour, self.previous_bbox, self.confidence

    def _predict_with_optical_flow(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict contour position using optical flow.

        Args:
            current_frame (np.ndarray): Current frame

        Returns:
            Optional[np.ndarray]: Predicted contour
        """
        if self.previous_contour is None or len(self.previous_contour) == 0:
            return None

        method = self.config['optical_flow_method']

        if method == 'lucas_kanade':
            return self._lucas_kanade_track(current_frame)
        elif method == 'farneback':
            return self._farneback_track(current_frame)
        else:
            logger.warning(f"Unknown optical flow method: {method}")
            return None

    def _lucas_kanade_track(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Track using Lucas-Kanade optical flow.

        Args:
            current_frame (np.ndarray): Current frame

        Returns:
            Optional[np.ndarray]: Tracked contour
        """
        # Extract points from contour
        if len(self.previous_contour.shape) == 3:
            points = self.previous_contour.squeeze().astype(np.float32)
        else:
            points = self.previous_contour.astype(np.float32)

        points = points.reshape(-1, 1, 2)

        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            current_frame,
            points,
            None,
            winSize=self.config['lk_win_size'],
            maxLevel=self.config['lk_max_level'],
            criteria=self.config['lk_criteria']
        )

        # Keep only good points
        if new_points is not None:
            good_new = new_points[status == 1]
            if len(good_new) > 3:  # Need at least 3 points for a contour
                return good_new.reshape(-1, 1, 2).astype(np.int32)

        return None

    def _farneback_track(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Track using Farneback dense optical flow.

        Args:
            current_frame (np.ndarray): Current frame

        Returns:
            Optional[np.ndarray]: Tracked contour
        """
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_frame,
            current_frame,
            None,
            self.config['fb_pyr_scale'],
            self.config['fb_levels'],
            self.config['fb_winsize'],
            self.config['fb_iterations'],
            self.config['fb_poly_n'],
            self.config['fb_poly_sigma'],
            0
        )

        # Apply flow to contour points
        if len(self.previous_contour.shape) == 3:
            points = self.previous_contour.squeeze()
        else:
            points = self.previous_contour

        new_points = []
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                dx, dy = flow[y, x]
                new_x = x + dx
                new_y = y + dy
                new_points.append([new_x, new_y])

        if len(new_points) > 3:
            return np.array(new_points, dtype=np.int32).reshape(-1, 1, 2)

        return None

    def _predict_with_kalman(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Predict bbox using Kalman filter.

        Returns:
            Optional[Tuple]: Predicted bbox
        """
        if self.kalman is None:
            return None

        # Predict
        prediction = self.kalman.predict()

        # Extract bbox
        x = int(prediction[0])
        y = int(prediction[1])
        w = int(prediction[2])
        h = int(prediction[3])

        return (x, y, w, h)

    def _fuse_predictions(self, detected_contour: Optional[np.ndarray],
                         detected_bbox: Optional[Tuple],
                         predicted_contour: Optional[np.ndarray],
                         kalman_bbox: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """
        Fuse detection and prediction.

        Args:
            detected_contour: Detected contour
            detected_bbox: Detected bbox
            predicted_contour: Predicted contour
            kalman_bbox: Kalman predicted bbox

        Returns:
            Tuple of (fused_contour, fused_bbox)
        """
        # If detection available and confident, use it
        if detected_contour is not None and self.confidence > self.config['min_confidence']:
            fused_contour = detected_contour
        elif predicted_contour is not None:
            fused_contour = predicted_contour
        else:
            fused_contour = self.previous_contour

        # Fuse bbox
        if detected_bbox is not None and kalman_bbox is not None:
            # Average detection and prediction
            fused_bbox = tuple([int((d + k) / 2) for d, k in zip(detected_bbox, kalman_bbox)])
        elif detected_bbox is not None:
            fused_bbox = detected_bbox
        elif kalman_bbox is not None:
            fused_bbox = kalman_bbox
        else:
            fused_bbox = self.previous_bbox

        return fused_contour, fused_bbox

    def _validate_tracking(self, contour: Optional[np.ndarray], bbox: Optional[Tuple]) -> bool:
        """
        Validate tracking results.

        Args:
            contour: Tracked contour
            bbox: Tracked bbox

        Returns:
            bool: True if valid
        """
        if contour is None or bbox is None:
            return False

        # Check displacement
        if self.previous_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = self.previous_bbox
            curr_x, curr_y, curr_w, curr_h = bbox

            # Compute center displacement
            prev_cx = prev_x + prev_w / 2
            prev_cy = prev_y + prev_h / 2
            curr_cx = curr_x + curr_w / 2
            curr_cy = curr_y + curr_h / 2

            displacement = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)

            if displacement > self.config['max_displacement_per_frame']:
                logger.warning(f"Excessive displacement: {displacement}")
                return False

        # Check area change
        if self.previous_area is not None:
            current_area = cv2.contourArea(contour)
            area_change_ratio = abs(current_area - self.previous_area) / (self.previous_area + 1e-6)

            if area_change_ratio > self.config['max_area_change_percent']:
                logger.warning(f"Excessive area change: {area_change_ratio}")
                return False

        return True

    def _update_state(self, frame: np.ndarray, contour: Optional[np.ndarray],
                     bbox: Optional[Tuple]):
        """
        Update tracker state.

        Args:
            frame: Current frame
            contour: Current contour
            bbox: Current bbox
        """
        self.previous_frame = frame.copy()
        self.previous_contour = contour.copy() if contour is not None else None
        self.previous_bbox = bbox

        if contour is not None:
            self.previous_area = cv2.contourArea(contour)
            self.area_history.append(self.previous_area)
            self.contour_history.append(contour.copy())

        if bbox is not None:
            self.bbox_history.append(bbox)

            # Update Kalman filter
            if self.kalman is not None:
                self._update_kalman(bbox)

    def _initialize_kalman(self) -> cv2.KalmanFilter:
        """
        Initialize Kalman filter.

        Returns:
            cv2.KalmanFilter: Kalman filter
        """
        # State: [x, y, w, h, dx, dy, dw, dh]
        kalman = cv2.KalmanFilter(8, 4)

        # Transition matrix
        kalman.transitionMatrix = np.eye(8, dtype=np.float32)
        kalman.transitionMatrix[0, 4] = 1  # x += dx
        kalman.transitionMatrix[1, 5] = 1  # y += dy
        kalman.transitionMatrix[2, 6] = 1  # w += dw
        kalman.transitionMatrix[3, 7] = 1  # h += dh

        # Measurement matrix
        kalman.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        kalman.measurementMatrix[0, 0] = 1  # measure x
        kalman.measurementMatrix[1, 1] = 1  # measure y
        kalman.measurementMatrix[2, 2] = 1  # measure w
        kalman.measurementMatrix[3, 3] = 1  # measure h

        # Process noise
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * self.config['process_noise']

        # Measurement noise
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * self.config['measurement_noise']

        return kalman

    def _init_kalman_state(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize Kalman filter state.

        Args:
            bbox: Initial bounding box
        """
        if self.kalman is None:
            return

        x, y, w, h = bbox
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)

    def _update_kalman(self, bbox: Tuple[int, int, int, int]):
        """
        Update Kalman filter with measurement.

        Args:
            bbox: Measured bounding box
        """
        if self.kalman is None:
            return

        measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
        self.kalman.correct(measurement)

    def get_smoothed_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get smoothed trajectory of bbox centers.

        Returns:
            List[Tuple[int, int]]: List of (cx, cy) coordinates
        """
        trajectory = []
        for bbox in self.bbox_history:
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + h // 2
            trajectory.append((cx, cy))

        return trajectory

    def get_area_over_time(self) -> List[float]:
        """
        Get area measurements over time.

        Returns:
            List[float]: Area values
        """
        return list(self.area_history)

    def detect_compression_event(self) -> Dict[str, any]:
        """
        Detect compression/release events based on area changes.

        Returns:
            Dict: Event information
        """
        if len(self.area_history) < 3:
            return {'compression_detected': False}

        areas = list(self.area_history)

        # Compute area changes
        changes = np.diff(areas)

        # Detect rapid decrease (compression)
        compression_threshold = -0.15  # 15% decrease
        release_threshold = 0.10  # 10% increase

        recent_change = changes[-1] / (areas[-2] + 1e-6)

        event_info = {
            'compression_detected': recent_change < compression_threshold,
            'release_detected': recent_change > release_threshold,
            'area_change_ratio': recent_change,
            'current_area': areas[-1],
            'min_area': min(areas),
            'max_area': max(areas),
            'compression_ratio': min(areas) / (max(areas) + 1e-6)
        }

        return event_info

    def reset(self):
        """Reset tracker state."""
        self.previous_frame = None
        self.previous_contour = None
        self.previous_bbox = None
        self.previous_area = None
        self.confidence = 1.0
        self.contour_history.clear()
        self.area_history.clear()
        self.bbox_history.clear()

        if self.config['use_kalman']:
            self.kalman = self._initialize_kalman()


def create_temporal_tracker(config: Optional[Dict] = None) -> TemporalTracker:
    """
    Factory function to create temporal tracker.

    Args:
        config (Optional[Dict]): Configuration

    Returns:
        TemporalTracker: Configured tracker
    """
    return TemporalTracker(config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from video_loader import VideoLoader
        from preprocessing import UltrasoundPreprocessor
        from roi_detector import ROIDetector
        from boundary_detectors import create_canny_detector

        video_path = sys.argv[1]

        with VideoLoader(video_path) as loader:
            preprocessor = UltrasoundPreprocessor()
            roi_detector = ROIDetector()
            detector = create_canny_detector()
            tracker = TemporalTracker()

            # Process first few frames
            for frame_num in range(min(10, loader.metadata['frame_count'])):
                frame = loader.extract_frame(frame_num)
                preprocessed = preprocessor.preprocess(frame)
                roi, bbox = roi_detector.detect_roi(preprocessed)

                # Detect
                edges = detector.detect_combined(roi)
                contour = detector.get_dominant_boundary(edges)

                # Track
                if contour is not None:
                    tracked_contour, tracked_bbox, confidence = tracker.track(roi, contour, bbox)
                    print(f"Frame {frame_num}: confidence={confidence:.2f}")

            # Get statistics
            event_info = tracker.detect_compression_event()
            print(f"Compression event info: {event_info}")
    else:
        print("Usage: python temporal_tracker.py <video_path>")
