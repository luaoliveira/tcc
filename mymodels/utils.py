from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
from argparse import ArgumentParser

def save_outputs_as_masks(outputs, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for i, output in enumerate(outputs):

        mask=(output.squeeze().cpu().numpy() > 0.5).astype('uint8')
        mask_image= Image.fromarray(mask*255)
        mask_image.save(os.path.join(output_dir, f"mask_{i}.png"))



"""
remove the padding from the image
"""
def remove_padding(mask_pil):
    # Crop the mask back to 400x400 by removing 8px padding on each side
    cropped_mask = mask_pil.crop((8, 8, 408, 408))  # (left, top, right, bottom)

    return cropped_mask

"""
plot the image and the mask side by side in groups of 2
"""

def plot_images_with_masks(original_folder, mask_folder, model_name):
    original_images = os.listdir(original_folder)
    mask_images = os.listdir(mask_folder)
    
    # Set the number of images to plot
    num_images = 5
    #min(len(original_images), len(mask_images))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))

    for i in range(num_images):
        original_img_path = os.path.join(original_folder, original_images[i])
        mask_img_path = os.path.join(mask_folder, mask_images[i])

        # Open images
        original_image = Image.open(original_img_path).convert("RGB")
        mask_image = Image.open(mask_img_path).convert("L")

        # Plot the original image
        axes[i, 0].imshow(original_image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original Images')

        # Plot the mask
        axes[i, 1].imshow(mask_image, cmap='gray')  # Use gray for the mask
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Masks')

    plt.tight_layout()
    plt.savefig(f'{model_name}_vs_original_imgs_comp.png')


# def plot_images_side_by_side(original_folder, mask_folder):
#     original_images = os.listdir(original_folder)
#     mask_images = os.listdir(mask_folder)
    
#     # Set the number of images to plot
#     num_images = min(len(original_images), len(mask_images))
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(1, 4, figsize=(15, 5))

#     for i in range(2):  # Only process the first two images for a single row
#         original_img_path = os.path.join(original_folder, original_images[i])
#         mask_img_path = os.path.join(mask_folder, mask_images[i])

#         # Open the images
#         original_image = Image.open(original_img_path).convert("RGB")
#         mask_image = Image.open(mask_img_path).convert("L")

#         # Plot the original image
#         axes[i * 2].imshow(original_image)
#         axes[i * 2].axis('off')
#         axes[i * 2].set_title(f'Original Image {i+1}')

#         # Plot the mask
#         axes[i * 2 + 1].imshow(mask_image, cmap='gray', alpha=0.6)
#         axes[i * 2 + 1].axis('off')
#         axes[i * 2 + 1].set_title(f'Mask {i+1}')

#     plt.tight_layout()
    
#     # Save the plot
#     # plt.savefig(save_path)
#     plt.show()


def plot_grid_image_masks(original_folder, mask_folder, save_path='results.png'):
    original_images = os.listdir(original_folder)
    mask_images = os.listdir(mask_folder)

    num_images = min(len(original_images), len(mask_images))
    cols = 4  # 2 original + 2 mask in each row (pair of image and mask side by side)
    rows = num_images // 2 + num_images % 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))

    for i in range(num_images):
        original_img_path = os.path.join(original_folder, original_images[i])
        mask_img_path = os.path.join(mask_folder, mask_images[i])

        # Open the images
        original_image = Image.open(original_img_path).convert("RGB")
        mask_image = Image.open(mask_img_path).convert("L")
        inverted_mask = Image.eval(mask_image, lambda x: 255 - x)

        # Row and column positioning
        row_idx = i // 2
        col_idx = (i % 2) * 2

        # Plot the original image
        axes[row_idx, col_idx].imshow(original_image)
        axes[row_idx, col_idx].axis('off')
        axes[row_idx, col_idx].set_title(f'Image {i+1}')

        # Plot the mask
        axes[row_idx, col_idx + 1].imshow(inverted_mask, cmap='Grays', alpha=1)
        axes[row_idx, col_idx + 1].axis('off')
        axes[row_idx, col_idx + 1].set_title(f'Mask {i+1}')

    plt.tight_layout()

    # Save the plot
    plt.subplots_adjust(wspace=0.02, hspace=-0.6)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def overlay_mask(img_path, mask_path, color=(0, 0, 255), pad=(8,8), opacity=0.3):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    # img.convert("RGBA")

    color_mask = Image.new("RGB", img.size, color)

    box = (pad[0], pad[1], mask.size[0] - pad[0], mask.size[1] - pad[1])
    mask = mask.crop(box)
    color_mask = Image.composite(color_mask, img, mask)

    # print(color_mask.size, img.size, mask.size)

    img = img.convert(color_mask.mode)

    return Image.blend(img, color_mask, opacity)

def plot_error_curves(train_losses, val_losses, file_name, model_name):

    plt.figure(figsize=(10,10))
    plt.plot(train_losses, label='Erro de treino')
    plt.plot(val_losses, label='Erro de validação')

    # Increase font size of legend labels
    plt.legend(fontsize=16)  # Adjust fontsize as needed

    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Erros', fontsize=16)
    plt.title(f'Curvas de erro {model_name}', fontsize=16)

    plt.savefig(f'{file_name}.png')

def parse_args() -> dict :
    
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML file containing the training parameters")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f, yaml.SafeLoader)
    
    return configs