"""
Video Loader Module

Handles video file loading, frame extraction, and metadata storage for ultrasound videos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoLoader:
    """
    Loads and processes ultrasound video files.

    Attributes:
        video_path (Path): Path to the video file
        metadata (Dict): Video metadata (fps, resolution, frame count)
        cap (cv2.VideoCapture): OpenCV video capture object
    """

    def __init__(self, video_path: str, downsample_factor: float = 1.0):
        """
        Initialize the video loader.

        Args:
            video_path (str): Path to the video file
            downsample_factor (float): Factor to downsample frames (1.0 = no downsampling)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.downsample_factor = downsample_factor
        self.cap = None
        self.metadata = {}

        # Initialize video capture
        self._load_video()

    def _load_video(self):
        """Load video and extract metadata."""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        # Extract metadata
        self.metadata = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'downsampled_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.downsample_factor),
            'downsampled_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.downsample_factor)
        }

        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Metadata: {self.metadata}")

    def get_metadata(self) -> Dict:
        """
        Get video metadata.

        Returns:
            Dict: Video metadata
        """
        return self.metadata.copy()

    def extract_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from the video.

        Args:
            frame_number (int): Frame number to extract (0-indexed)

        Returns:
            np.ndarray: Grayscale frame or None if extraction fails
        """
        if frame_number >= self.metadata['frame_count']:
            logger.warning(f"Frame number {frame_number} exceeds total frames {self.metadata['frame_count']}")
            return None

        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_number}")
            return None

        return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame: convert to grayscale and downsample if needed.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed grayscale frame
        """
        # Convert to grayscale if not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Downsample if needed
        if self.downsample_factor != 1.0:
            new_width = self.metadata['downsampled_width']
            new_height = self.metadata['downsampled_height']
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return gray

    def extract_all_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields all frames from the video.

        Yields:
            Tuple[int, np.ndarray]: (frame_number, frame)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        frame_number = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self._process_frame(frame)
            yield frame_number, processed_frame
            frame_number += 1

    def extract_frames_batch(self, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
        """
        Extract a batch of frames.

        Args:
            start_frame (int): Starting frame number
            end_frame (Optional[int]): Ending frame number (exclusive). None = all frames

        Returns:
            np.ndarray: Array of frames with shape (num_frames, height, width)
        """
        if end_frame is None:
            end_frame = self.metadata['frame_count']

        num_frames = end_frame - start_frame
        frames = []

        for i in range(start_frame, end_frame):
            frame = self.extract_frame(i)
            if frame is not None:
                frames.append(frame)

        return np.array(frames)

    def sample_frames(self, sample_rate: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Sample frames at a specified rate.

        Args:
            sample_rate (int): Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)

        Yields:
            Tuple[int, np.ndarray]: (frame_number, frame)
        """
        for frame_number, frame in self.extract_all_frames():
            if frame_number % sample_rate == 0:
                yield frame_number, frame

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract frame at a specific time.

        Args:
            time_seconds (float): Time in seconds

        Returns:
            np.ndarray: Frame at specified time or None if invalid
        """
        frame_number = int(time_seconds * self.metadata['fps'])
        return self.extract_frame(frame_number)

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Released video: {self.video_path.name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()


def load_video(video_path: str, downsample_factor: float = 1.0) -> VideoLoader:
    """
    Convenience function to load a video.

    Args:
        video_path (str): Path to video file
        downsample_factor (float): Downsample factor

    Returns:
        VideoLoader: Loaded video object
    """
    return VideoLoader(video_path, downsample_factor)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

        with VideoLoader(video_path) as loader:
            print(f"Video metadata: {loader.get_metadata()}")

            # Extract first frame
            first_frame = loader.extract_frame(0)
            if first_frame is not None:
                print(f"First frame shape: {first_frame.shape}")
                print(f"First frame dtype: {first_frame.dtype}")
                print(f"First frame range: [{first_frame.min()}, {first_frame.max()}]")
    else:
        print("Usage: python video_loader.py <video_path>")
