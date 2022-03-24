# Виконати 2D лінійну фільтрацію зображення з різними значеннями ядра. Провести
# порівняльний аналіз

import matplotlib.pyplot as plt
import matplotlib.image as Image
import numpy as np


def plot(img1, img2, image_type, kernel_type):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Original ' + image_type)
    ax[0].axis('off')

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title('With ' + kernel_type)
    ax[1].axis('off')

    plt.show()


def img_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def add_padding(img, padding_size):
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_size * 2,
        img.shape[1] + padding_size * 2
    ))

    img_with_padding[padding_size:-padding_size,
                     padding_size:-padding_size] = img

    return img_with_padding


def get_output_img_size(img_size, kernel_size):
    output_img_size = 0

    for i in range(img_size):
        count = i + kernel_size
        if count <= img_size:
            output_img_size += 1

    return output_img_size


def apply_filter2D(image, kernel):
    kernel_size = kernel.shape[0]
    padding_size = kernel_size // 2
    img_with_padding = add_padding(image, padding_size)

    output_size = get_output_img_size(
        img_size=img_with_padding.shape[0],
        kernel_size=kernel_size
    )

    k = kernel_size
    convolved_img = np.zeros(shape=(output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            mat = img_with_padding[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


# define kernels

# blur filter
blur_kernel = np.ones((10, 10), np.float32)/25

# edge detection
edge_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Sobel filter
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernels = [[blur_kernel, 'blur_kernel'], [
    edge_kernel, 'edge_kernel'], [sobel_kernel, 'sobel_kernel']]

images = ['high_contrast.jpeg', 'low_contrast.jpeg',
          'high_detailed.jpeg', 'low_detailed.jpeg']

for image_name in images:
    for kernel in kernels:
        # image
        image = np.array(img_to_gray(Image.imread(image_name)))
        # filtered img
        filtered_img = apply_filter2D(image=image, kernel=kernel[0])
        # plot result
        plot(img1=image, img2=filtered_img,
             image_type=image_name, kernel_type=kernel[1])
