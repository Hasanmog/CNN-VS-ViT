def file_format_counter(imgs_paths):
    '''
    simple function to count the number of images corresponding to each of the following extensions : .png , .jpg and .bmp
    Arguments:
    imgs_paths : list of the imgs paths
    returns : number of .png , .jpg and .bmp images in this list
    '''
    png , jpg , bmp = 0 , 0 , 0
    for index,img in enumerate(imgs_paths):
        if img.endswith('.png'):
            png+=1
        elif img.endswith(".jpg"):
            jpg+=1
        else:
            bmp +=1
            
    return png , jpg , bmp