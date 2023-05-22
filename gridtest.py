import numpy as np

# Example list of (x, y) values
xy_values = [(1, 2), (3, 4), (5, 6), (7, 8)]

# Extract x and y values from the list
x_values = [xy[0] for xy in xy_values]
y_values = [xy[1] for xy in xy_values]

# Find the minimum and maximum values of x and y
min_x = min(x_values)
max_x = max(x_values)
min_y = min(y_values)
max_y = max(y_values)

# Create a grid of coordinates using numpy
grid = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=int)

# Assign the values to their corresponding coordinates in the grid
for x, y in xy_values:
    grid[y - min_y, x - min_x] = 1  # Assign 1 to indicate the presence of the coordinate

# Print the resulting grid
print(grid)
