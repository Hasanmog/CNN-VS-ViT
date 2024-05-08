import random
import os

def dataset_split(img_dir, masks_dir):
    # List all image files
    images = os.listdir(img_dir)
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(images)  # Shuffle images

    # Split into train, test, and validation sets
    train = images[:280]
    test = images[280:340]
    val = images[340:]

    sets = [train, test, val]
    train_masks = []
    test_masks = []
    val_masks = []
    masks_sets = [train_masks, test_masks, val_masks]

    # List all mask files
    masks = os.listdir(masks_dir)
    mask_dict = {mask.split('.')[0]: mask for mask in masks}  # Create a dictionary with numerical part as key

    # Match masks to images in each set
    for i, img_set in enumerate(sets):
        for image in img_set:
            num_id = image.split('.')[0]  # Extract the numerical ID from the image filename
            corresponding_mask = mask_dict.get(num_id, None)  # Get the corresponding mask using the numerical ID
            if corresponding_mask:
                masks_sets[i].append(corresponding_mask)

    return sets, masks_sets