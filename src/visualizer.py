"""
Visualizer Module

Generates visualizations and outputs for boundary detection results.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import logging

logger = logging.getLogger(__name__)


class BoundaryVisualizer:
    """
    Visualizes and exports boundary detection results.
    """

    def __init__(self, output_dir: str, config: Optional[Dict] = None):
        """
        Initialize the visualizer.

        Args:
            output_dir (str): Output directory for results
            config (Optional[Dict]): Configuration parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            # Video output
            'output_fps': 30,
            'output_codec': 'mp4v',
            'boundary_color': (0, 255, 0),  # Green
            'boundary_thickness': 2,
            'show_roi': True,
            'roi_color': (255, 0, 0),  # Red
            'roi_thickness': 1,

            # Confidence visualization
            'show_confidence': True,
            'low_confidence_color': (0, 0, 255),  # Red
            'high_confidence_color': (0, 255, 0),  # Green

            # Metrics overlay
            'show_metrics': True,
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 0.5,
            'font_thickness': 1,
            'text_color': (255, 255, 255),

            # Compression events
            'highlight_compression': True,
            'compression_color': (255, 165, 0),  # Orange

            # Plot settings
            'plot_dpi': 100,
            'plot_figsize': (12, 8)
        }

        if config:
            self.config.update(config)

        logger.info(f"Visualizer initialized with output dir: {output_dir}")

    def create_annotated_video(self, video_path: str, results: List[Dict],
                               output_name: str = "annotated_video.mp4"):
        """
        Create annotated video with detected boundaries.

        Args:
            video_path (str): Path to original video
            results (List[Dict]): List of frame results
            output_name (str): Output video filename
        """
        from video_loader import VideoLoader

        output_path = self.output_dir / output_name

        with VideoLoader(video_path) as loader:
            # Get video properties
            fps = self.config['output_fps']
            frame_width = loader.metadata['width']
            frame_height = loader.metadata['height']

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config['output_codec'])
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

            try:
                for result in results:
                    frame_num = result['frame_number']
                    frame = loader.extract_frame(frame_num)

                    if frame is None:
                        continue

                    # Annotate frame
                    annotated = self.annotate_frame(frame, result)

                    # Write
                    out.write(annotated)

                logger.info(f"Annotated video saved to: {output_path}")

            finally:
                out.release()

    def annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Annotate a single frame with boundary and metadata.

        Args:
            frame (np.ndarray): Input frame
            result (Dict): Frame result dictionary

        Returns:
            np.ndarray: Annotated frame
        """
        # Convert to BGR if grayscale
        if len(frame.shape) == 2:
            annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            annotated = frame.copy()

        # Draw ROI
        if self.config['show_roi'] and 'roi_bbox' in result:
            self._draw_roi(annotated, result['roi_bbox'])

        # Draw boundary
        if 'contour' in result and result['contour'] is not None:
            color = self._get_boundary_color(result)
            self._draw_contour(annotated, result['contour'], result.get('roi_bbox'), color)

        # Add metrics overlay
        if self.config['show_metrics']:
            self._draw_metrics(annotated, result)

        # Highlight compression events
        if self.config['highlight_compression'] and result.get('compression_event'):
            self._highlight_compression(annotated)

        return annotated

    def _draw_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Draw ROI rectangle.

        Args:
            image (np.ndarray): Image to draw on
            bbox (Tuple): Bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h),
                     self.config['roi_color'],
                     self.config['roi_thickness'])

    def _draw_contour(self, image: np.ndarray, contour: np.ndarray,
                     roi_bbox: Optional[Tuple], color: Tuple[int, int, int]):
        """
        Draw boundary contour.

        Args:
            image (np.ndarray): Image to draw on
            contour (np.ndarray): Contour points
            roi_bbox (Optional[Tuple]): ROI bbox to offset contour
            color (Tuple): BGR color
        """
        # Offset contour if ROI bbox provided
        if roi_bbox is not None:
            x_offset, y_offset, _, _ = roi_bbox
            contour_offset = contour.copy()
            contour_offset[:, :, 0] += x_offset
            contour_offset[:, :, 1] += y_offset
        else:
            contour_offset = contour

        # Draw contour
        cv2.drawContours(image, [contour_offset], 0, color,
                        self.config['boundary_thickness'])

    def _get_boundary_color(self, result: Dict) -> Tuple[int, int, int]:
        """
        Get boundary color based on confidence.

        Args:
            result (Dict): Frame result

        Returns:
            Tuple[int, int, int]: BGR color
        """
        if not self.config['show_confidence'] or 'confidence' not in result:
            return self.config['boundary_color']

        confidence = result['confidence']

        # Interpolate between low and high confidence colors
        low_color = np.array(self.config['low_confidence_color'])
        high_color = np.array(self.config['high_confidence_color'])

        color = low_color + confidence * (high_color - low_color)
        return tuple(color.astype(int).tolist())

    def _draw_metrics(self, image: np.ndarray, result: Dict):
        """
        Draw metrics overlay on image.

        Args:
            image (np.ndarray): Image to draw on
            result (Dict): Frame result
        """
        metrics = result.get('metrics', {})
        frame_num = result.get('frame_number', 0)

        # Prepare text lines
        lines = [
            f"Frame: {frame_num}",
        ]

        if 'area' in metrics:
            lines.append(f"Area: {metrics['area']:.1f} px")

        if 'confidence' in result:
            lines.append(f"Confidence: {result['confidence']:.2f}")

        if 'circularity' in metrics:
            lines.append(f"Circularity: {metrics['circularity']:.2f}")

        # Draw text
        y_offset = 30
        for line in lines:
            cv2.putText(image, line, (10, y_offset),
                       self.config['font'],
                       self.config['font_scale'],
                       self.config['text_color'],
                       self.config['font_thickness'])
            y_offset += 25

    def _highlight_compression(self, image: np.ndarray):
        """
        Highlight compression event on frame.

        Args:
            image (np.ndarray): Image to highlight
        """
        # Draw colored border
        cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1),
                     self.config['compression_color'], 5)

        # Add text
        cv2.putText(image, "COMPRESSION", (image.shape[1] - 200, 30),
                   self.config['font'], 0.7,
                   self.config['compression_color'], 2)

    def _convert_to_serializable(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif obj is None:
            return None
        else:
            return obj

    def export_metrics_csv(self, results: List[Dict], filename: str = "metrics.csv"):
        """
        Export metrics to CSV file.

        Args:
            results (List[Dict]): List of frame results
            filename (str): Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as csvfile:
            # Determine all possible metric keys
            all_keys = set()
            for result in results:
                if 'metrics' in result:
                    all_keys.update(result['metrics'].keys())

            # Create header
            fieldnames = ['frame_number', 'confidence'] + sorted(list(all_keys))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write data
            for result in results:
                row = {
                    'frame_number': result.get('frame_number', 0),
                    'confidence': result.get('confidence', 0.0)
                }

                if 'metrics' in result:
                    row.update(result['metrics'])

                writer.writerow(row)

        logger.info(f"Metrics CSV saved to: {output_path}")

    def export_boundaries_json(self, results: List[Dict], filename: str = "boundaries.json"):
        """
        Export boundary data to JSON.

        Args:
            results (List[Dict]): List of frame results
            filename (str): Output filename
        """
        output_path = self.output_dir / filename

        # Convert numpy arrays and types to native Python types for JSON serialization
        json_data = []
        for result in results:
            result_copy = self._convert_to_serializable(result)
            json_data.append(result_copy)

        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Boundaries JSON saved to: {output_path}")

    def plot_area_over_time(self, results: List[Dict], filename: str = "area_plot.png"):
        """
        Plot vessel area over time.

        Args:
            results (List[Dict]): List of frame results
            filename (str): Output filename
        """
        output_path = self.output_dir / filename

        # Extract data
        frames = []
        areas = []

        for result in results:
            if 'metrics' in result and 'area' in result['metrics']:
                frames.append(result['frame_number'])
                areas.append(result['metrics']['area'])

        # Create plot
        plt.figure(figsize=self.config['plot_figsize'], dpi=self.config['plot_dpi'])
        plt.plot(frames, areas, 'b-', linewidth=2)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Vessel Area (pixels)', fontsize=12)
        plt.title('Vessel Area Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Highlight compression events
        for result in results:
            if result.get('compression_event'):
                frame = result['frame_number']
                if frame in frames:
                    idx = frames.index(frame)
                    plt.axvline(x=frame, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Area plot saved to: {output_path}")

    def plot_metrics_comparison(self, results: List[Dict],
                               metrics: List[str] = ['area', 'circularity'],
                               filename: str = "metrics_comparison.png"):
        """
        Plot multiple metrics for comparison.

        Args:
            results (List[Dict]): List of frame results
            metrics (List[str]): List of metric names to plot
            filename (str): Output filename
        """
        output_path = self.output_dir / filename

        # Extract data
        frames = [r['frame_number'] for r in results]
        metric_data = {m: [] for m in metrics}

        for result in results:
            if 'metrics' in result:
                for metric in metrics:
                    value = result['metrics'].get(metric, 0)
                    metric_data[metric].append(value)
            else:
                for metric in metrics:
                    metric_data[metric].append(0)

        # Create subplots
        fig, axes = plt.subplots(len(metrics), 1,
                                figsize=(self.config['plot_figsize'][0],
                                        self.config['plot_figsize'][1] * len(metrics) / 2),
                                dpi=self.config['plot_dpi'])

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            axes[idx].plot(frames, metric_data[metric], linewidth=2)
            axes[idx].set_xlabel('Frame Number')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Over Time')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Metrics comparison plot saved to: {output_path}")

    def save_key_frames(self, video_path: str, results: List[Dict],
                       num_frames: int = 5, filename_prefix: str = "keyframe"):
        """
        Save key frames with boundaries.

        Args:
            video_path (str): Path to original video
            results (List[Dict]): List of frame results
            num_frames (int): Number of key frames to save
            filename_prefix (str): Prefix for saved files
        """
        from video_loader import VideoLoader

        # Select evenly spaced frames
        indices = np.linspace(0, len(results) - 1, num_frames, dtype=int)

        with VideoLoader(video_path) as loader:
            for idx in indices:
                result = results[idx]
                frame_num = result['frame_number']
                frame = loader.extract_frame(frame_num)

                if frame is None:
                    continue

                # Annotate
                annotated = self.annotate_frame(frame, result)

                # Save
                output_path = self.output_dir / f"{filename_prefix}_{frame_num:04d}.png"
                cv2.imwrite(str(output_path), annotated)

        logger.info(f"Saved {num_frames} key frames")

    def create_summary_report(self, results: List[Dict], video_metadata: Dict,
                             filename: str = "summary_report.txt"):
        """
        Create a text summary report.

        Args:
            results (List[Dict]): List of frame results
            video_metadata (Dict): Video metadata
            filename (str): Output filename
        """
        output_path = self.output_dir / filename

        # Compute statistics
        areas = [r['metrics']['area'] for r in results if 'metrics' in r and 'area' in r['metrics']]
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        compression_events = sum(1 for r in results if r.get('compression_event', False))

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ULTRASOUND BOUNDARY DETECTION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("VIDEO INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Frames: {video_metadata.get('frame_count', 0)}\n")
            f.write(f"FPS: {video_metadata.get('fps', 0):.2f}\n")
            f.write(f"Resolution: {video_metadata.get('width', 0)} x {video_metadata.get('height', 0)}\n\n")

            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Frames Processed: {len(results)}\n")
            f.write(f"Compression Events Detected: {compression_events}\n\n")

            if areas:
                f.write("VESSEL AREA STATISTICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean Area: {np.mean(areas):.2f} pixels\n")
                f.write(f"Std Area: {np.std(areas):.2f} pixels\n")
                f.write(f"Min Area: {np.min(areas):.2f} pixels\n")
                f.write(f"Max Area: {np.max(areas):.2f} pixels\n")
                f.write(f"Compression Ratio: {np.min(areas) / np.max(areas):.3f}\n\n")

            if confidences:
                f.write("TRACKING CONFIDENCE\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean Confidence: {np.mean(confidences):.3f}\n")
                f.write(f"Min Confidence: {np.min(confidences):.3f}\n\n")

            f.write("=" * 60 + "\n")

        logger.info(f"Summary report saved to: {output_path}")


def create_visualizer(output_dir: str, config: Optional[Dict] = None) -> BoundaryVisualizer:
    """
    Factory function to create visualizer.

    Args:
        output_dir (str): Output directory
        config (Optional[Dict]): Configuration

    Returns:
        BoundaryVisualizer: Configured visualizer
    """
    return BoundaryVisualizer(output_dir, config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Create a simple test visualization
        output_dir = sys.argv[1]
        visualizer = BoundaryVisualizer(output_dir)

        # Create dummy results
        results = []
        for i in range(100):
            results.append({
                'frame_number': i,
                'confidence': 0.8 + 0.2 * np.sin(i / 10),
                'metrics': {
                    'area': 5000 + 1000 * np.sin(i / 20),
                    'circularity': 0.8 + 0.1 * np.cos(i / 15)
                },
                'compression_event': i % 30 == 0
            })

        # Generate plots
        visualizer.plot_area_over_time(results)
        visualizer.plot_metrics_comparison(results)
        visualizer.export_metrics_csv(results)

        print(f"Visualizations created in {output_dir}")
    else:
        print("Usage: python visualizer.py <output_directory>")
