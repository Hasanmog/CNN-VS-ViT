def scale_to_unit_range(tensor):
    tensor = tensor - tensor.min()  # Shift the minimum value to 0.0
    tensor = tensor / tensor.max()  # Scale the maximum value to 1.0
    return tensor