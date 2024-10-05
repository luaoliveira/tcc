def save_outputs_as_masks(outputs, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for i, output in enumerate(outputs):

        mask=(output.squeeze().cpu().numpy() > 0.5).astype('uint8')
        mask_image= Image.fromarray(mask*255)
        mask_image.save(os.path.join(output_dir, f"mask_{i}.png"))
