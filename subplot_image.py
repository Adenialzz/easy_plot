import time
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def read_images(path_list):
    images = []
    for p in path_list:
        image = cv2.imread(p)
        images.append(image[:, :, ::-1])
    return images


def main():

    # 读取输入、输出、gt 图片。
    inp_images_path_list = glob('./viz_images/source*.png')
    out_images_path_list = glob('./viz_images/denoised*.png')
    gt_images_path_list = glob('./viz_images/reference*.png')
    srt_key = lambda x: int(x.split('.')[1].split('_e')[-1])  # ./viz_images/source_e1.png

    inp_images_path_list.sort(key=srt_key)
    out_images_path_list.sort(key=srt_key)
    gt_images_path_list.sort(key=srt_key)

    print(inp_images_path_list)
    print(out_images_path_list)
    print(gt_images_path_list)
    input_images = read_images(inp_images_path_list)
    output_images = read_images(out_images_path_list)
    gt_images = read_images(gt_images_path_list)
    n_epochs = len(input_images)

      # Create subplots
    interval = 4
    n_rows = n_epochs // interval
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))  # Adjust figsize for better visualization

    for i in range(0, n_rows):
        ep = i * interval
        epoch_str = f"ep={ep}"
        # axes[i, 0].annotate(f"ep={ep}", xy=(-1.0, 0.5), xycoords='axes fraction', fontsize=14)

        # Input image
        axes[i, 0].imshow(input_images[ep])
        axes[i, 0].set_title(f"{epoch_str} - Input", fontsize=6)
        axes[i, 0].axis('off')

        # Output image
        axes[i, 1].imshow(output_images[ep])
        axes[i, 1].set_title(f"{epoch_str} - Output", fontsize=6)
        axes[i, 1].axis('off')

        # Ground truth image
        axes[i, 2].imshow(gt_images[ep])
        axes[i, 2].set_title(f"{epoch_str} - GT", fontsize=6)
        axes[i, 2].axis('off')

    # for ax in axes.flatten():
    #     ax.set_frame_on(False)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig('viz_images.png', bbox_inches='tight', dpi=350)

if __name__ == "__main__":
    main()

