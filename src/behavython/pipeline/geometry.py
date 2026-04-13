from shapely.geometry import Polygon
import math
import numpy as np


def angle_between_lines(line1, line2, origin):
    """
    angle_between_lines Returns the angle between two lines given an origin

    Args:
        line1 (tuple): Two tuples containing the coordinates of the start and end points of the first line
        line2 (tuple): Two tuples containing the coordinates of the start and end points of the second line
        origin (tuple): A tuple containing the coordinates of the origin of the lines

    Returns:
        float: The angle between the two lines in degrees with the origin as the vertex
    """
    # Line segments are represented by tuples of two points
    # (x1, y1) -> start point of line segment 1
    # (x2, y2) -> end point of line segment 1
    # (x3, y3) -> start point of line segment 2
    # (x4, y4) -> end point of line segment 2
    x, y = origin

    x1, y1 = line1[0][0] - x, line1[0][1] - y
    x2, y2 = line1[1][0] - x, line1[1][1] - y
    x3, y3 = line2[0][0] - x, line2[0][1] - y
    x4, y4 = line2[1][0] - x, line2[1][1] - y

    # Calculate the dot product of the two vectors
    dot_product = x1 * x3 + y1 * y3 + x2 * x4 + y2 * y4

    # Calculate the magnitudes of the two vectors
    magnitude_a = math.sqrt(x1**2 + y1**2 + x2**2 + y2**2)
    magnitude_b = math.sqrt(x3**2 + y3**2 + x4**2 + y4**2)

    # Calculate the angle in radians
    angle = math.acos(dot_product / (magnitude_a * magnitude_b))

    # Convert to degrees and return
    return math.degrees(angle)


def line_trough_triangle_vertex(A, B, C):
    """
    line_trough_triangle_vertex Returns the points of a line passing through the center of the triangle's `A` vertex

    Args:
        A (tuple): A tuple containing the coordinates of the `A` vertex
        B (tuple): A tuple containing the coordinates of the `B` vertex
        C (tuple): A tuple containing the coordinates of the `C` vertex

    Returns:
        tuple: A tuple containing the coordinates of the start and end points of the line
    """

    # Compute the midpoint of the side opposite vertex A
    M = (B + C) / 2
    # Compute the vector from vertex A to the midpoint
    AM = M - A
    # Define the endpoints of the line passing through the center of vertex A
    P = A - 0.5 * AM
    Q = A + 0.1 * AM

    return P, Q


def sign(x):
    return -1 if x < 0 else 1


def detect_collision(line_segment_start, line_segment_end, circle_center, circle_radius):
    """Detects intersections between a line segment and a circle.

    Args:
        line_start (tuple): A tuple containing the (x, y) coordinates of the start point of the line segment.
        line_end (tuple): A tuple containing the (x, y) coordinates of the end point of the line segment.
        circle_center (tuple): A tuple containing the (x, y) coordinates of the center of the circle.
        circle_radius (float): The radius of the circle.

    Returns:
        list: A list of tuples representing the intersection points between the line segment and the circle.

    """

    line_start_x_relative_to_circle = line_segment_start[0] - circle_center[0]
    line_start_y_relative_to_circle = line_segment_start[1] - circle_center[1]
    line_end_x_relative_to_circle = line_segment_end[0] - circle_center[0]
    line_end_y_relative_to_circle = line_segment_end[1] - circle_center[1]
    line_segment_delta_x = line_end_x_relative_to_circle - line_start_x_relative_to_circle
    line_segment_delta_y = line_end_y_relative_to_circle - line_start_y_relative_to_circle

    line_segment_length = math.sqrt(line_segment_delta_x * line_segment_delta_x + line_segment_delta_y * line_segment_delta_y)
    discriminant_numerator = (
        line_start_x_relative_to_circle * line_end_y_relative_to_circle - line_end_x_relative_to_circle * line_start_y_relative_to_circle
    )
    discriminant = circle_radius * circle_radius * line_segment_length * line_segment_length - discriminant_numerator * discriminant_numerator
    if discriminant < 0:
        return []
    if discriminant == 0:
        intersection_point_1_x = (discriminant_numerator * line_segment_delta_y) / (line_segment_length * line_segment_length)
        intersection_point_1_y = (-discriminant_numerator * line_segment_delta_x) / (line_segment_length * line_segment_length)
        parameterization_a = (intersection_point_1_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
            intersection_point_1_y - line_start_y_relative_to_circle
        ) * line_segment_delta_y / line_segment_length
        return (
            [(intersection_point_1_x + circle_center[0], intersection_point_1_y + circle_center[1])]
            if 0 < parameterization_a < line_segment_length
            else []
        )

    intersection_point_1_x = (
        discriminant_numerator * line_segment_delta_y + sign(line_segment_delta_y) * line_segment_delta_x * math.sqrt(discriminant)
    ) / (line_segment_length * line_segment_length)
    intersection_point_1_y = (-discriminant_numerator * line_segment_delta_x + abs(line_segment_delta_y) * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    parameterization_a = (intersection_point_1_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
        intersection_point_1_y - line_start_y_relative_to_circle
    ) * line_segment_delta_y / line_segment_length
    intersection_points = (
        [(intersection_point_1_x + circle_center[0], intersection_point_1_y + circle_center[1])]
        if 0 < parameterization_a < line_segment_length
        else []
    )

    intersection_point_2_x = (
        discriminant_numerator * line_segment_delta_y - sign(line_segment_delta_y) * line_segment_delta_x * math.sqrt(discriminant)
    ) / (line_segment_length * line_segment_length)
    intersection_point_2_y = (-discriminant_numerator * line_segment_delta_x - abs(line_segment_delta_y) * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    parameterization_b = (intersection_point_2_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
        intersection_point_2_y - line_start_y_relative_to_circle
    ) * line_segment_delta_y / line_segment_length
    intersection_points += (
        [(intersection_point_2_x + circle_center[0], intersection_point_2_y + circle_center[1])]
        if 0 < parameterization_b < line_segment_length
        else []
    )
    return intersection_points


def is_inside_circle(x, y, circle_X, circle_Y, circle_D):
    """
    Determines if a given point (x, y) is inside a circle defined by a region of interest (roi_X, roi_Y, roi_D).
    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        circle_X (float): The x-coordinate of the circle center.
        circle_Y (float): The y-coordinate of the circle center.
        circle_D (float): The diameter of the circle.
    Returns:
        bool: True if the point is inside the circle, False otherwise.
    """
    # Calculate the radius
    radius = circle_D / 2.0

    # Calculate the distance between the point and the circle center
    distance = math.sqrt((x - circle_X) ** 2 + (y - circle_Y) ** 2)

    # Check if the distance is less than or equal to the radius
    return distance <= radius


def calculate_triangle_area(pt1, pt2, pt3):
    """
    Calculates the area of a triangle given its three vertices using Heron's formula.
    Used to calculate the mice's head area.

    Args:
        pt1 (tuple/list): Coordinates of the first point (e.g., focinho)
        pt2 (tuple/list): Coordinates of the second point (e.g., orelha_esq)
        pt3 (tuple/list): Coordinates of the third point (e.g., orelha_dir)

    Returns:
        float: Area of the triangle.
    """
    side1 = math.sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))
    side2 = math.sqrt(((pt3[0] - pt2[0]) ** 2) + ((pt3[1] - pt2[1]) ** 2))
    side3 = math.sqrt(((pt1[0] - pt3[0]) ** 2) + ((pt1[1] - pt3[1]) ** 2))

    s = (side1 + side2 + side3) / 2

    # max(0, ...) ensures float precision issues don't cause a negative square root
    area_squared = s * (s - side1) * (s - side2) * (s - side3)
    return math.sqrt(max(0, area_squared))


def calculate_gaze_angle_and_distance(P, Q, A, T):
    """
    Calculates the angle between the animal's gaze vector and the target ROI,
    as well as the Euclidean distance to the target.

    Args:
        P (tuple/array): Start point of the gaze line
        Q (tuple/array): End point of the gaze line
        A (tuple/array): Position of the animal's nose (origin of target vector)
        T (tuple/array): Position of the ROI center

    Returns:
        tuple: (angle_in_degrees, distance_to_target)
    """
    # Gaze vector (P -> Q)
    v_gaze = np.array(Q) - np.array(P)

    # Target vector (nose -> ROI)
    v_target = np.array(T) - np.array(A)

    gaze_norm = np.linalg.norm(v_gaze)
    target_norm = np.linalg.norm(v_target)

    # Prevent division by zero if vectors are zero-length
    if gaze_norm == 0 or target_norm == 0:
        return 0.0, target_norm

    # Normalize
    v_gaze_n = v_gaze / gaze_norm
    v_target_n = v_target / target_norm

    # Angle (degrees)
    cos_theta = np.clip(np.dot(v_gaze_n, v_target_n), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    # Distance to ROI center
    distance = target_norm

    return angle_deg, distance


def determine_interaction_state(angle_deg, distance, prev_distance):
    """
    Determines the interaction state (approaching, retreating, looking, neutral)
    based on the gaze angle and change in distance.

    Args:
        angle_deg (float): Angle to the target in degrees.
        distance (float): Current distance to the target.
        prev_distance (float | None): Previous frame's distance to the target.

    Returns:
        str: The state of interaction.
    """
    state = "neutral"

    if angle_deg <= 90:
        if prev_distance is not None and distance < prev_distance:
            state = "approaching"
        else:
            state = "looking"
    else:
        if prev_distance is not None and distance > prev_distance:
            state = "retreating"

    return state


def create_frequency_grid(x_values, y_values, bin_size, analysis_range, speed=None, mean_speed=None):
    """
    Creates a frequency grid (heatmap) based on the given x and y values.
    Optimized with NumPy vectorization.

    Args:
        x_values (np.ndarray): Array of x-coordinate values.
        y_values (np.ndarray): Array of y-coordinate values.
        bin_size (float): Size of each bin in the grid.
        analysis_range (tuple): Range of indices (start, end) to consider for analysis.
        speed (np.ndarray, optional): Array of speed values corresponding to coordinates.
        mean_speed (float, optional): Threshold speed to filter the data.

    Returns:
        np.ndarray: The frequency grid of shape (num_bins_y, num_bins_x).
    """
    start, end = analysis_range
    x = np.asarray(x_values)[start:end]
    y = np.asarray(y_values)[start:end]

    # Apply speed filter if provided
    if speed is not None and mean_speed is not None:
        # Assuming the speed array provided is already sliced, or slice it if it matches the full length
        if len(speed) == len(x_values):
            speed = speed[start:end]

        mask = speed > mean_speed
        x = x[mask]
        y = y[mask]

    # Handle edge case where no data remains
    if len(x) == 0 or len(y) == 0:
        return np.array([[]], dtype=int)

    # Calculate exact bin edges based on the min/max of the data
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # np.arange goes up to, but doesn't include the stop value, so we add a buffer
    bins_x = np.arange(min_x, max_x + bin_size, bin_size)
    bins_y = np.arange(min_y, max_y + bin_size, bin_size)

    # np.histogram2d natively maps coordinates into a 2D frequency grid.
    # We pass (y, x) to ensure the output matrix shape is (rows=y, cols=x) to match your original logic.
    grid, _, _ = np.histogram2d(y, x, bins=[bins_y, bins_x])

    return grid.astype(int)

def build_plus_maze_geometry(points: list[tuple[float, float]]) -> dict[str, Polygon]:
    """
    Builds the 5 standard polygons for the elevated plus maze.
    Expects exactly 12 points outlining the maze structure in a continuous loop.
    
    Mapping based on standard reference points:
    points[0:4]   -> Top Open Arm (Top-Left, Top-Right, Center-Top-Right, Center-Top-Left)
    points[3:7]   -> Right Closed Arm (Center-Top-Right, Top-Right, Bottom-Right, Center-Bottom-Right)
    points[6:10]  -> Bottom Open Arm (Center-Bottom-Right, Bottom-Right, Bottom-Left, Center-Bottom-Left)
    points[9:12] + points[0] -> Left Closed Arm
    """
    if len(points) != 12:
        raise ValueError(f"Elevated Plus Maze requires exactly 12 points, got {len(points)}")

    return {
        "top_open": Polygon(points[0:4]),
        "right_closed": Polygon(points[3:7]),
        "bottom_open": Polygon(points[6:10]),
        "left_closed": Polygon(points[9:12] + [points[0]]),
        "center": Polygon([points[3], points[6], points[9], points[0]])
    }

def build_grid_open_field_geometry(
    corners: list[tuple[float, float]], 
    grid_size: tuple[int, int] = (3, 3)
) -> dict[str, Polygon]:
    """
    Mathematically interpolates an NxM grid from 4 outer corners.
    Expected corners clicking order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    if len(corners) != 4:
        raise ValueError(f"Open Field grid requires exactly 4 corners, got {len(corners)}")

    rows, cols = grid_size
    tl, tr, br, bl = [np.array(pt) for pt in corners]
    
    grid_polygons = {}
    
    for row in range(rows):
        for col in range(cols):
            y_frac_top = row / rows
            y_frac_bottom = (row + 1) / rows
            x_frac_left = col / cols
            x_frac_right = (col + 1) / cols
            
            cell_tl = tl + (tr - tl) * x_frac_left + (bl - tl) * y_frac_top
            cell_tr = tl + (tr - tl) * x_frac_right + (bl - tl) * y_frac_top
            cell_br = tl + (tr - tl) * x_frac_right + (bl - tl) * y_frac_bottom
            cell_bl = tl + (tr - tl) * x_frac_left + (bl - tl) * y_frac_bottom
            
            zone_name = f"zone_r{row}_c{col}"
            grid_polygons[zone_name] = Polygon([cell_tl, cell_tr, cell_br, cell_bl])
            
    return grid_polygons