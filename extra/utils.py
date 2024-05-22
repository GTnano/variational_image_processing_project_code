import numpy as np


def batch_images(images: list = None, num_rows: int = 10, num_cols: int = 10) -> np.ndarray:
    """batch images

    ### Args:
        images (list, optional): list of images. Defaults to [].
        num_rows (int, optional): number of images in a row. Defaults to 10.
        num_cols (int, optional): number of imagess in a col. Defaults to 10.
    """
    if len(images) == 0:
        print("No images found. exit..")
        return np.zeros((1, 1))  # Return an empty array if no images are found

    if len(images) % (num_cols * num_rows) != 0:
        print("Number of images is not divisible by the number of images in a row and column. exit..")
        return np.zeros((1, 1))

    height, width = images[0].shape[:2]
    comb_img = np.zeros((height * num_rows, width * num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(images):
                img = images[index]
                comb_img[i * height: (i + 1) * height, j * width: (j + 1) * width] = img

    return comb_img


def image_size(img):
    height, width = img.shape[:2]
    if height > width:
        return "vertical"
    else:
        return "horizontal"


def create_img_list(images):
    vertical_images = []
    horizontal_images = []

    for img in images:
        size = image_size(img)
        if size == "vertical":
            vertical_images.append(img)
        else:
            horizontal_images.append(img)
    return vertical_images, horizontal_images