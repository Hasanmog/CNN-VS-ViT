def scale_to_unit_range(tensor):
    tensor = tensor - tensor.min()  # Shift the minimum value to 0.0
    tensor = tensor / tensor.max()  # Scale the maximum value to 1.0
    return tensor

def format_for_display(images):
    
    formatted_images  = []
    for image in images:
        image = image.permute(1 , 2 , 0)
        image = scale_to_unit_range(image)
        image = image.numpy()
        formatted_images.append(image)
        
    return formatted_images