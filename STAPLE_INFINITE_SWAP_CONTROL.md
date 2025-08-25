# STAPLE Infinite Swap Control

## Overview

The STAPLE implementation now includes graceful control over infinite swap functionality to prevent weird results that can occur in STAPLE simulations. By default, infinite swap is **disabled** for STAPLE, similar to how REPPTIS works.

## Default Behavior (Infinite Swap Disabled)

When infinite swap is disabled (default):
- Each path stays in its own ensemble (100% probability)
- No path swapping occurs between ensembles
- Returns an identity matrix for the probability calculation
- Ghost ensemble has zero probability
- This prevents problematic matrix calculations that can occur with STAPLE's contiguous structure requirements

## Configuration

### To keep infinite swap DISABLED (recommended for STAPLE):
```toml
[simulation]
# No need to specify anything - disabled by default
seed = 42
steps = 10000
```

Or explicitly:
```toml
[simulation]
seed = 42
steps = 10000
staple_infinite_swap = false  # Explicitly disable (same as default)
```

### To ENABLE infinite swap:
```toml
[simulation]
seed = 42
steps = 10000
staple_infinite_swap = true  # Enable infinite swap for STAPLE
```

## Implementation Details

### Code Changes
1. **New config parameter**: `staple_infinite_swap` in the `[simulation]` section
2. **Default value**: `false` (disabled)
3. **Property override**: The `prob` property in `REPEX_state_staple` checks this flag
4. **Logging**: Informative messages when swap mode is determined

### Behavior Comparison

| Mode | Probability Matrix | Path Behavior | Use Case |
|------|-------------------|---------------|----------|
| Disabled (default) | Identity matrix | No swaps between ensembles | Standard STAPLE simulations |
| Enabled | Full permanent calculation | Full infinite swap | Testing/debugging |

### Matrix Structure

When **disabled**:
```
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 0.]]  # Ghost ensemble = 0
```

When **enabled**:
- Uses full `inf_retis` calculation
- Complex matrix with off-diagonal elements
- Allows path swapping between ensembles

## Easy Removal

To remove this functionality later (if needed):

1. Delete the `staple_infinite_swap` parameter from `__init__`
2. Remove the `prob` property override
3. STAPLE will then inherit the base REPEX behavior (infinite swap enabled)

## Logging

The implementation includes logging to make the current mode clear:
- `"STAPLE: Infinite swap DISABLED - using identity matrix (no path swaps)"`
- `"STAPLE: Infinite swap ENABLED - using full permanent calculation"`

## Benefits

1. **Prevents weird results**: Avoids problematic matrix calculations in STAPLE
2. **Graceful configuration**: Easy to toggle via config file
3. **Clear logging**: Makes the current mode obvious
4. **Easy removal**: Can be removed later without major changes
5. **Similar to REPPTIS**: Follows established pattern for disabling infinite swap

## Example Usage

For typical STAPLE simulations, use the default (no configuration needed):

```bash
# Your existing STAPLE simulation will automatically use disabled infinite swap
python your_staple_simulation.py config.toml
```

For debugging/testing with infinite swap enabled:

```toml
# config_with_swap.toml
[simulation]
staple_infinite_swap = true
# ... other settings
```

```bash
python your_staple_simulation.py config_with_swap.toml
```
