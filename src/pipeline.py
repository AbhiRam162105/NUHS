"""
Ultrasound Boundary Detection Pipeline

Main orchestration pipeline for traditional CV-based boundary detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from tqdm import tqdm
import time

from video_loader import VideoLoader
from preprocessing import UltrasoundPreprocessor
from roi_detector import ROIDetector
from boundary_detectors import (
    create_canny_detector,
    create_active_contour_detector,
    create_watershed_detector,
    create_level_set_detector
)
from boundary_refiner import BoundaryRefiner
from temporal_tracker import TemporalTracker
from visualizer import BoundaryVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltrasoundBoundaryPipeline:
    """
    Complete pipeline for ultrasound vessel boundary detection.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config_path (Optional[str]): Path to configuration YAML file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self._initialize_components()

        logger.info("Pipeline initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path (Optional[str]): Path to config file

        Returns:
            Dict: Configuration dictionary
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {config_path}")
        else:
            # Default configuration
            config = {
                'preprocessing': {},
                'roi_detection': {},
                'boundary_detection': {
                    'methods': ['canny', 'active_contour', 'watershed'],  # Methods to use
                    'canny': {},
                    'active_contour': {},
                    'watershed': {},
                    'level_set': {}
                },
                'refinement': {},
                'tracking': {},
                'visualization': {},
                'output': {
                    'directory': 'data/output',
                    'save_video': True,
                    'save_metrics': True,
                    'save_plots': True,
                    'save_keyframes': True
                },
                'processing': {
                    'sample_rate': 1,  # Process every Nth frame
                    'max_frames': None,  # None = all frames
                    'use_multiprocessing': False
                }
            }
            logger.info("Using default configuration")

        return config

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Preprocessor
        self.preprocessor = UltrasoundPreprocessor(self.config.get('preprocessing'))

        # ROI Detector
        self.roi_detector = ROIDetector(self.config.get('roi_detection'))

        # Boundary Detectors
        self.detectors = {}
        methods = self.config['boundary_detection'].get('methods', ['canny'])

        if 'canny' in methods:
            self.detectors['canny'] = create_canny_detector(
                self.config['boundary_detection'].get('canny')
            )

        if 'active_contour' in methods:
            self.detectors['active_contour'] = create_active_contour_detector(
                self.config['boundary_detection'].get('active_contour')
            )

        if 'watershed' in methods:
            self.detectors['watershed'] = create_watershed_detector(
                self.config['boundary_detection'].get('watershed')
            )

        if 'level_set' in methods:
            self.detectors['level_set'] = create_level_set_detector(
                self.config['boundary_detection'].get('level_set')
            )

        # Boundary Refiner
        self.refiner = BoundaryRefiner(self.config.get('refinement'))

        # Temporal Tracker
        self.tracker = TemporalTracker(self.config.get('tracking'))

        # Visualizer
        output_dir = self.config['output']['directory']
        self.visualizer = BoundaryVisualizer(output_dir, self.config.get('visualization'))

        logger.info(f"Initialized {len(self.detectors)} boundary detectors: {list(self.detectors.keys())}")

    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process entire video.

        Args:
            video_path (str): Path to video file

        Returns:
            List[Dict]: List of frame results
        """
        logger.info(f"Processing video: {video_path}")

        results = []
        sample_rate = self.config['processing']['sample_rate']
        max_frames = self.config['processing']['max_frames']

        with VideoLoader(video_path) as loader:
            video_metadata = loader.get_metadata()
            total_frames = min(video_metadata['frame_count'], max_frames) if max_frames else video_metadata['frame_count']

            # Process frames
            with tqdm(total=total_frames // sample_rate, desc="Processing frames") as pbar:
                for frame_num, frame in loader.sample_frames(sample_rate):
                    if max_frames and frame_num >= max_frames:
                        break

                    # Process frame
                    result = self.process_frame(frame, frame_num)
                    results.append(result)

                    pbar.update(1)

            logger.info(f"Processed {len(results)} frames")

            # Generate outputs
            if self.config['output']['save_video']:
                logger.info("Creating annotated video...")
                self.visualizer.create_annotated_video(video_path, results)

            if self.config['output']['save_metrics']:
                logger.info("Exporting metrics...")
                self.visualizer.export_metrics_csv(results)
                self.visualizer.export_boundaries_json(results)

            if self.config['output']['save_plots']:
                logger.info("Generating plots...")
                self.visualizer.plot_area_over_time(results)
                self.visualizer.plot_metrics_comparison(results, ['area', 'circularity'])

            if self.config['output']['save_keyframes']:
                logger.info("Saving key frames...")
                self.visualizer.save_key_frames(video_path, results)

            # Create summary report
            self.visualizer.create_summary_report(results, video_metadata)

        return results

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Process a single frame.

        Args:
            frame (np.ndarray): Input frame
            frame_number (int): Frame number

        Returns:
            Dict: Frame result
        """
        result = {
            'frame_number': frame_number,
            'timestamp': time.time()
        }

        try:
            # Step 1: Preprocessing
            preprocessed = self.preprocessor.preprocess(frame)

            # Step 2: ROI Detection
            roi, roi_bbox = self.roi_detector.detect_roi(preprocessed)
            result['roi_bbox'] = roi_bbox

            # Step 3: Boundary Detection (multiple methods)
            boundaries = {}

            if 'canny' in self.detectors:
                edges = self.detectors['canny'].detect_combined(roi)
                boundaries['canny'] = edges

            if 'active_contour' in self.detectors:
                # Use edge initialization if available
                if 'canny' in boundaries:
                    contour = self.detectors['active_contour'].detect_with_edge_initialization(roi, boundaries['canny'])
                else:
                    contour = self.detectors['active_contour'].detect(roi)

                # Convert to binary mask
                mask = self.detectors['active_contour'].contour_to_mask(contour, roi.shape)
                boundaries['active_contour'] = mask

            if 'watershed' in self.detectors:
                labels, vessel_mask, _ = self.detectors['watershed'].detect_full_pipeline(roi)
                boundaries['watershed'] = vessel_mask

            if 'level_set' in self.detectors:
                segmentation = self.detectors['level_set'].detect(roi)
                boundaries['level_set'] = segmentation

            # Step 4: Boundary Refinement
            refined, metrics = self.refiner.ensemble_refine(boundaries, roi)

            # Extract contour from refined boundary
            contours = self.refiner._extract_contours(refined)
            if len(contours) > 0:
                contour = self.refiner._select_best_contour(contours)
            else:
                contour = None

            # Step 5: Temporal Tracking
            tracked_contour, tracked_bbox, confidence = self.tracker.track(roi, contour, roi_bbox)

            result['contour'] = tracked_contour
            result['confidence'] = confidence
            result['metrics'] = metrics

            # Detect compression events
            event_info = self.tracker.detect_compression_event()
            result['compression_event'] = event_info.get('compression_detected', False)
            result['event_info'] = event_info

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            result['error'] = str(e)
            result['contour'] = None
            result['confidence'] = 0.0
            result['metrics'] = {}

        return result

    def process_frame_batch(self, frames: List[np.ndarray], start_frame_num: int) -> List[Dict]:
        """
        Process a batch of frames (for parallel processing).

        Args:
            frames (List[np.ndarray]): List of frames
            start_frame_num (int): Starting frame number

        Returns:
            List[Dict]: List of results
        """
        results = []
        for i, frame in enumerate(frames):
            result = self.process_frame(frame, start_frame_num + i)
            results.append(result)
        return results

    def benchmark_detectors(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Benchmark different detectors on a single frame.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            Dict[str, float]: Detector timings in seconds
        """
        preprocessed = self.preprocessor.preprocess(frame)
        roi, _ = self.roi_detector.detect_roi(preprocessed)

        timings = {}

        for name, detector in self.detectors.items():
            start = time.time()

            try:
                if name == 'canny':
                    _ = detector.detect_combined(roi)
                elif name == 'active_contour':
                    _ = detector.detect(roi)
                elif name == 'watershed':
                    _ = detector.detect_full_pipeline(roi)
                elif name == 'level_set':
                    _ = detector.detect(roi)

                elapsed = time.time() - start
                timings[name] = elapsed

            except Exception as e:
                logger.error(f"Error benchmarking {name}: {e}")
                timings[name] = -1

        return timings

    def save_config(self, output_path: str):
        """
        Save current configuration to YAML file.

        Args:
            output_path (str): Output path
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Configuration saved to: {output_path}")


def main():
    """Main entry point for pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Ultrasound Boundary Detection Pipeline")
    parser.add_argument('video', type=str, help="Path to input video file")
    parser.add_argument('--config', type=str, default=None, help="Path to configuration YAML file")
    parser.add_argument('--output-dir', type=str, default='data/output', help="Output directory")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmark on first frame")

    args = parser.parse_args()

    # Update output directory if specified
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Create pipeline
    pipeline = UltrasoundBoundaryPipeline(args.config)

    # Override output directory if specified
    if args.output_dir:
        pipeline.config['output']['directory'] = args.output_dir
        pipeline.visualizer = BoundaryVisualizer(args.output_dir, pipeline.config.get('visualization'))

    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running benchmark...")
        with VideoLoader(args.video) as loader:
            frame = loader.extract_frame(0)
            timings = pipeline.benchmark_detectors(frame)

            print("\nDetector Benchmarks:")
            print("-" * 40)
            for name, timing in timings.items():
                print(f"{name:20s}: {timing:.3f} seconds")
            print("-" * 40)

    # Process video
    logger.info("Starting video processing...")
    start_time = time.time()

    results = pipeline.process_video(args.video)

    elapsed_time = time.time() - start_time

    logger.info(f"Processing complete!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average FPS: {len(results) / elapsed_time:.2f}")
    logger.info(f"Results saved to: {pipeline.config['output']['directory']}")


if __name__ == "__main__":
    main()
