# Traditional Boundary Detection Pipeline for Ultrasound Videos

A comprehensive computer vision pipeline for detecting and tracking vessel boundaries in B-mode ultrasound videos using classical (non-deep learning) image processing techniques.

## Overview

This pipeline implements multiple traditional computer vision algorithms to detect vessel boundaries in ultrasound videos:

- **Edge-based detection** (Canny, Sobel)
- **Active contours** (Snakes)
- **Watershed segmentation**
- **Level set methods** (Chan-Vese, Geodesic Active Contours)

The pipeline combines results from multiple detectors, refines boundaries, and tracks them temporally across video frames using optical flow and Kalman filtering.

## Features

- Multiple boundary detection algorithms with ensemble fusion
- Temporal tracking with optical flow and Kalman filtering
- Speckle noise reduction optimized for ultrasound imaging
- ROI detection to focus computational effort
- Compression event detection
- Comprehensive visualization and metrics export
- Fully configurable via YAML files

## Project Structure

```
.
├── src/
│   ├── video_loader.py           # Video I/O and frame extraction
│   ├── preprocessing.py           # Noise reduction and enhancement
│   ├── roi_detector.py            # Region of Interest detection
│   ├── boundary_detectors/
│   │   ├── canny_detector.py      # Edge-based detection
│   │   ├── active_contour.py      # Snake-based detection
│   │   ├── watershed_detector.py  # Watershed segmentation
│   │   └── level_set.py           # Level set methods
│   ├── boundary_refiner.py        # Multi-method fusion and refinement
│   ├── temporal_tracker.py        # Frame-to-frame tracking
│   ├── visualizer.py              # Output generation
│   └── pipeline.py                # Main orchestration
├── config/
│   └── params.yaml                # Configuration file
├── data/
│   ├── input/                     # Input videos
│   └── output/                    # Results
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd NUHS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process an ultrasound video with default settings:

```bash
python src/pipeline.py data/input/your_video.mp4
```

### With Custom Configuration

```bash
python src/pipeline.py data/input/your_video.mp4 --config config/params.yaml --output-dir data/output/my_results
```

### Benchmark Mode

Test detector performance on the first frame:

```bash
python src/pipeline.py data/input/your_video.mp4 --benchmark
```

## Configuration

All pipeline parameters can be configured via `config/params.yaml`. Key sections include:

### Preprocessing
```yaml
preprocessing:
  median_kernel_size: 5
  gaussian_sigma: 1.2
  clahe_clip_limit: 2.0
  morphological_enabled: true
```

### Boundary Detection
```yaml
boundary_detection:
  methods:
    - canny
    - active_contour
    - watershed
  # Individual detector parameters...
```

### Temporal Tracking
```yaml
tracking:
  optical_flow_method: 'lucas_kanade'
  use_kalman: true
  max_displacement_per_frame: 25
```

See `config/params.yaml` for complete configuration options.

## Pipeline Stages

### 1. Video Loading & Frame Extraction
- Loads `.mp4` video files
- Converts to grayscale
- Optional downsampling for performance

### 2. Preprocessing
- **Speckle reduction**: Median filter + Gaussian blur
- **Contrast enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Morphological preprocessing**: Top-hat and black-hat transforms

### 3. ROI Detection
- Automatic vessel region localization using Otsu's thresholding
- Filters regions by size, circularity, and aspect ratio
- Reduces computational load by processing only relevant regions

### 4. Boundary Detection
Multiple methods run in parallel:

- **Canny Edge Detection**: Adaptive thresholding based on image statistics
- **Active Contours (Snakes)**: Energy-minimizing curves that conform to boundaries
- **Watershed Segmentation**: Region-based segmentation using distance transforms
- **Level Set Methods**: Implicit contour evolution (optional, slower)

### 5. Boundary Refinement
- **Fusion**: Combines results from multiple detectors using voting or weighted averaging
- **Morphological refinement**: Closing and opening operations to clean boundaries
- **Contour approximation**: Douglas-Peucker algorithm for smooth boundaries
- **Shape fitting**: Optional ellipse or circle fitting

### 6. Temporal Tracking
- **Optical Flow**: Lucas-Kanade or Farneback methods for point tracking
- **Kalman Filtering**: Smooth position and velocity estimates
- **Validation**: Checks for excessive displacement or area changes
- **Re-initialization**: Automatic recovery from tracking failures

### 7. Visualization & Export

**Outputs:**
- Annotated video with detected boundaries
- CSV file with frame-by-frame metrics
- JSON file with boundary coordinates
- Plots: Area over time, metrics comparison
- Key frame snapshots
- Summary report

## Output Files

After processing, the output directory contains:

```
data/output/
├── annotated_video.mp4       # Video with overlaid boundaries
├── metrics.csv               # Frame-by-frame quantitative data
├── boundaries.json           # Boundary coordinates in JSON
├── area_plot.png             # Vessel area vs. time
├── metrics_comparison.png    # Multi-metric plots
├── keyframe_0000.png         # Selected key frames
├── keyframe_0025.png
├── ...
└── summary_report.txt        # Text summary of results
```

## Metrics

The pipeline computes and exports:

- **Vessel area**: Contour area in pixels
- **Circularity**: 4π × area / perimeter²
- **Aspect ratio**: Major/minor axis ratio
- **Smoothness**: Curvature variation
- **Tracking confidence**: 0-1 score
- **Compression ratio**: min_area / max_area
- **Compression events**: Rapid area decreases

## Performance

Typical processing speeds (on CPU):

- **Canny**: ~10-20 ms/frame
- **Active Contour**: ~100-200 ms/frame
- **Watershed**: ~50-100 ms/frame
- **Level Set**: ~200-500 ms/frame

Full pipeline with 3 detectors: ~5-10 FPS on standard CPU.

## Parameter Tuning

### Quick Start Settings

For **high-quality vessels** (clear boundaries):
```yaml
boundary_detection:
  methods: [canny]  # Fast, single method
refinement:
  fusion_method: 'voting'
  voting_threshold: 1
```

For **low-quality vessels** (noisy, unclear boundaries):
```yaml
boundary_detection:
  methods: [canny, active_contour, watershed]  # Multiple methods
preprocessing:
  denoise_enabled: true  # Enable anisotropic diffusion
refinement:
  fusion_method: 'voting'
  voting_threshold: 2  # Require 2+ methods to agree
```

For **fast compression events**:
```yaml
tracking:
  max_displacement_per_frame: 50  # Increase tolerance
  max_area_change_percent: 0.5    # Allow larger changes
```

## Examples

### Process with Custom Sample Rate
```bash
# Process every 5th frame
python src/pipeline.py video.mp4 --config config/params.yaml
```
Then edit `params.yaml`:
```yaml
processing:
  sample_rate: 5
```

### Disable Specific Detectors
Edit `config/params.yaml`:
```yaml
boundary_detection:
  methods:
    - canny
    - active_contour
    # - watershed  # Disabled
    # - level_set  # Disabled
```

### Output Only Metrics (No Video)
```yaml
output:
  save_video: false
  save_metrics: true
  save_plots: true
  save_keyframes: false
```

## Troubleshooting

### Issue: No boundaries detected
- Check ROI detection settings (try `use_center_fallback: true`)
- Adjust preprocessing parameters (increase `clahe_clip_limit`)
- Lower circularity thresholds in ROI detection

### Issue: Tracking lost frequently
- Increase `max_displacement_per_frame`
- Increase `max_area_change_percent`
- Enable `reinit_on_lost: true`

### Issue: Noisy/jittery boundaries
- Enable temporal smoothing in tracking
- Increase Kalman filter `measurement_noise`
- Enable boundary smoothing in refinement

### Issue: Slow processing
- Disable `level_set` detector
- Reduce `active_contour.max_iterations`
- Increase `sample_rate` to skip frames

## Advanced Usage

### Using Individual Components

```python
from src.video_loader import VideoLoader
from src.preprocessing import UltrasoundPreprocessor
from src.roi_detector import ROIDetector
from src.boundary_detectors import create_canny_detector

# Load video
loader = VideoLoader('video.mp4')
frame = loader.extract_frame(0)

# Preprocess
preprocessor = UltrasoundPreprocessor()
preprocessed = preprocessor.preprocess(frame)

# Detect ROI
roi_detector = ROIDetector()
roi, bbox = roi_detector.detect_roi(preprocessed)

# Detect boundary
detector = create_canny_detector()
edges = detector.detect_combined(roi)
```

### Custom Detector Weights

For weighted fusion, adjust detector contributions:

```yaml
refinement:
  fusion_method: 'weighted_average'
  detector_weights:
    canny: 0.1          # Less weight on edges
    active_contour: 0.5 # More weight on active contours
    watershed: 0.4
```

## Testing

Run tests (if available):
```bash
pytest tests/
```

## Citation

If you use this pipeline in your research, please cite:

```
@software{ultrasound_boundary_pipeline,
  title = {Traditional Boundary Detection Pipeline for Ultrasound Videos},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/NUHS}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

This pipeline implements classical computer vision algorithms from:

- Canny, J. (1986). "A Computational Approach to Edge Detection"
- Kass, M., et al. (1988). "Snakes: Active contour models"
- Vincent, L., & Soille, P. (1991). "Watersheds in digital spaces"
- Chan, T. F., & Vese, L. A. (2001). "Active contours without edges"

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note**: This pipeline is designed for research purposes. For clinical applications, please ensure proper validation and regulatory compliance.
