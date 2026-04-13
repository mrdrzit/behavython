import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def apply_rolling_window_filter(data_x: pd.Series, data_y: pd.Series, window_size: int = 5, jump_threshold: int = 150) -> tuple[pd.Series, pd.Series]:
    """
    Applies a rolling window filter to remove tracking jumps that exceed the threshold,
    and performs a cubic spline interpolation to bridge the resulting gaps.
    """
    x_attrs = getattr(data_x, "attrs", {})
    y_attrs = getattr(data_y, "attrs", {})

    # Calculate rolling extremes
    rolling_max_x = data_x.rolling(window_size, closed="both").max()
    rolling_min_x = data_x.rolling(window_size, closed="both").min()
    rolling_max_y = data_y.rolling(window_size, closed="both").max()
    rolling_min_y = data_y.rolling(window_size, closed="both").min()

    # Find the indices where the rolling difference exceeds the jump threshold
    jump_indices_x = np.where((rolling_max_x - rolling_min_x) > jump_threshold)[0]
    jump_indices_y = np.where((rolling_max_y - rolling_min_y) > jump_threshold)[0]

    # Remove the jumps (masking as NaN)
    jumps_removed_x = data_x.copy()
    jumps_removed_x.iloc[jump_indices_x] = np.nan

    jumps_removed_y = data_y.copy()
    jumps_removed_y.iloc[jump_indices_y] = np.nan

    # Drop NaNs to prepare for interpolation
    valid_x = jumps_removed_x.dropna()
    valid_y = jumps_removed_y.dropna()

    # Generate cubic splines
    spline_x = CubicSpline(valid_x.index, valid_x.values, bc_type="clamped", extrapolate=False)
    spline_y = CubicSpline(valid_y.index, valid_y.values, bc_type="clamped", extrapolate=False)

    # Reconstruct continuous Series
    filtered_x = pd.Series(np.squeeze(spline_x(jumps_removed_x.index)), index=jumps_removed_x.index, name=data_x.name)
    filtered_y = pd.Series(np.squeeze(spline_y(jumps_removed_y.index)), index=jumps_removed_y.index, name=data_y.name)

    # Restore original pandas attributes if they exist
    filtered_x.attrs.update(x_attrs)
    filtered_y.attrs.update(y_attrs)

    return filtered_x, filtered_y
