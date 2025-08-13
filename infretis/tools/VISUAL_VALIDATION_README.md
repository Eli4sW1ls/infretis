# Visual Validation Configuration for INFRETIS STAPLE

## Overview

The visual validation functionality has been added to `REPEX_state_staple` to help debug and validate shooting moves in real-time. After every shooting move, the code will:

1. **Plot the trajectory path** with interfaces, shooting regions, and trajectory extremes
2. **Display detailed trajectory data** including path type, length, order parameters
3. **Pause execution** until you close the plot window
4. **Highlight shooting regions** (if they exist) and mark key trajectory features

## Configuration

Add this to your TOML configuration file to control visual validation:

```toml
[output]
visual_validation = true  # Enable visual validation (default: true)
# visual_validation = false  # Disable to run without plotting
```

## What You'll See

### 1. Detailed Trajectory Data Output
```
============================================================
TRAJECTORY DATA AFTER TREAT_OUTPUT - Path #12345
============================================================
             frac: [0. 0. 0. 0. 0.] (shape: (5,))
           max_op: 1.2
           min_op: -0.8
           length: 100
          weights: [first 5: [...], ..., last 5: [...]] (length: 100)
           adress: ['path_file1.xyz', 'path_file2.xyz']
     ens_save_idx: 0
            ptype: ABA
        sh_region: [20, 80]
============================================================
```

### 2. Interactive Plot Window
The plot displays:
- **Blue line**: Trajectory path (order parameter vs time)
- **Red/Orange lines**: Interface boundaries
- **Green shaded area**: Shooting region OP range (horizontal bands parallel to interfaces)
- **Thick green line**: Highlighted shooting region path segment for the specific ensemble
- **Green dashed lines**: Shooting region boundaries with annotations
- **Red triangle**: Maximum order parameter point
- **Purple triangle**: Minimum order parameter point
- **Gray dotted lines**: Additional interface boundaries
- **Info box**: Summary of trajectory properties

### 3. Ensemble-Specific Shooting Regions
The visualization handles shooting regions in dictionary format:
- **Dictionary format**: `sh_region = {ensemble_id: [start, end], ...}` - Shows only the shooting region for the current ensemble
- **Ensemble filtering**: Only displays the shooting region for the specific ensemble being visualized
- **Clear labeling**: Shooting region is labeled with the ensemble number for clarity
- **Missing data handling**: Shows available ensembles if the current ensemble has no shooting region data

### 3. Plot Features
- **Title**: Shows move type, path number, ensemble, length, and path type
- **Legend**: Explains all plot elements
- **Grid**: For better readability
- **Detailed info box**: Shows trajectory metadata

## Usage During Simulation

When running your INFRETIS simulation:

1. **Normal execution** continues until a shooting move occurs
2. **Automatic pause** when plot window opens
3. **Inspect the trajectory** visually to validate:
   - Proper interface crossings
   - Correct shooting region identification
   - Expected path behavior
   - Trajectory length and extremes
4. **Close plot window** to continue simulation
5. **Process repeats** for each subsequent shooting move

## Debugging Benefits

This feature helps you:
- **Validate shooting regions** are correctly identified
- **Check interface crossings** happen as expected
- **Verify path types** (ABA, BAB, etc.) are correct
- **Inspect trajectory quality** before acceptance/rejection
- **Debug unexpected behavior** in real-time
- **Understand ensemble transitions** visually

## Performance Impact

- **Minimal computational overhead** when disabled
- **Interactive debugging** when enabled (pauses execution)
- **Recommended for development/debugging** phases
- **Can be disabled for production** runs

## Example Integration

```python
# In your simulation script
config["output"]["visual_validation"] = True  # Enable validation

# Run simulation - plots will appear after shooting moves
# Close each plot window to continue execution
```

## Troubleshooting

If plots don't appear:
1. Check matplotlib installation: `pip install matplotlib`
2. Verify X11 forwarding if using SSH
3. Set `visual_validation = false` to disable if needed

The visual validation provides immediate feedback on trajectory quality and shooting move correctness, making it easier to debug and validate your STAPLE simulations.
