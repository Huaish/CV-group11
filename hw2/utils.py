import cv2
import numpy as np
from matplotlib import pyplot as plt

def save_image_grid(image_list, text_list, rows, cols, target_size=None, output_path='output/image_grid.jpg'):
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    axs = axs.reshape(rows, cols)

    for idx, (image, text) in enumerate(zip(image_list, text_list)):
        row = idx // cols
        col = idx % cols

        # Resize image
        if target_size is not None:
            image = cv2.resize(image, target_size)

        axs[row, col].imshow(image)

        axs[row, col].axis('off')
        axs[row, col].set_title(text, fontsize=16, color='black', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)