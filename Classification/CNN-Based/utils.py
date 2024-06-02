from typing import Any


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


# class conditional_resize(object):
#     # checks if images are not square , it makes it (h,h)
#     def __init__(self):
#         pass
        
#     def __call__(self , img):
        
#         h , w = img.size
#         if h!= 256:
#             h ==256
#             return img.resize((h , h))
#         if h != w :
#             return img.resize((h , h))
        
#         return img
    
#     def __repr__(self):
#         return self.__class__.__name__ + '()'

        
        
    