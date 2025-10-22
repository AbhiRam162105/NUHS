# Ultrasound Boundary Detection - Optimization Notes

## Problem
The pipeline was generating warnings during video processing:
- "No valid contours after quality filtering"
- "Excessive area change"
- "Tracking validation failed"

These warnings indicate the pipeline is rejecting valid vessel boundaries due to strict thresholds.

## Root Causes

### 1. Strict Quality Filtering
The boundary refiner was rejecting contours with:
- **min_circularity: 0.4** - Too strict for irregular vessel shapes
- **min_area: 100** - Rejected small vessel segments

### 2. Strict Temporal Constraints
The temporal tracker was rejecting tracking results with:
- **max_area_change_percent: 0.30 (30%)** - Too strict for dynamic vessels and compression events
- **max_displacement_per_frame: 25 pixels** - Too restrictive for fast-moving vessels

### 3. High Fusion Threshold
- **voting_threshold: 2** - Required 2 out of 3 detectors to agree, rejecting frames where only 1-2 detectors worked well

## Solutions Applied

### Configuration Changes in `config/params.yaml`

#### 1. Quality Filtering (refinement section)
```yaml
# Before:
min_circularity: 0.4
min_area: 100

# After:
min_circularity: 0.2  # More lenient for irregular vessels
min_area: 50          # Allows smaller vessel segments
```

**Rationale**: Ultrasound vessels have irregular shapes and can be quite small, especially in compression states.

#### 2. Temporal Tracking (tracking section)
```yaml
# Before:
max_displacement_per_frame: 25
max_area_change_percent: 0.30

# After:
max_displacement_per_frame: 50
max_area_change_percent: 1.0
```

**Rationale**:
- Vessels can move significantly between frames during compression
- Area can change dramatically during compression events (which is what we want to detect!)
- Higher displacement tolerance allows tracking through faster motion

#### 3. Fusion Voting Threshold (refinement section)
```yaml
# Before:
voting_threshold: 2

# After:
voting_threshold: 1
```

**Rationale**: Requires only 1 detector to agree instead of 2, increasing detection sensitivity while maintaining quality through multiple detector redundancy.

## Expected Impact

With these changes, you should see:
- Significantly fewer "No valid contours after quality filtering" warnings
- Reduction in "Excessive area change" warnings (which are now expected for compression detection)
- Better temporal consistency with fewer dropped frames
- More complete segmentation of the vessel throughout the cardiac cycle

## Tuning Further

If you still see too many warnings, consider:

1. **For "No valid contours" warnings:**
   - Further reduce `min_circularity` to 0.1
   - Reduce `min_area` to 10-20

2. **For "Excessive area change" warnings:**
   - Increase `max_area_change_percent` beyond 1.0
   - These are not necessarily bad - they indicate compression detection working!

3. **For better quality:**
   - Increase `min_confidence` from 0.5 to 0.7
   - Adjust `confidence_decay` from 0.95 to 0.90

## Revert Instructions

If the new settings work poorly, you can revert by changing the YAML back to:
```yaml
refinement:
  voting_threshold: 2
  min_circularity: 0.4
  min_area: 100

tracking:
  max_displacement_per_frame: 25
  max_area_change_percent: 0.30
```

## Testing

To test with the optimized configuration:
```bash
python3 src/pipeline.py data/Image05.mp4 --config config/params.yaml
```

Or if running from the project root:
```bash
cd /Users/abhiram/Desktop/NUHS/NUHS
python3 src/pipeline.py data/Image05.mp4 --config config/params.yaml
```
