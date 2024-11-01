import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix




def plot_confusion_matrix(true_labels, pred_labels, model_name):

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    labels = ['Foreground', 'Background']
    fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=labels,  # x-axis labels
    y=labels,  # y-axis labels
    colorscale="Blues",
    hoverongaps=False
    ))

    fig.show()
    fig.write_image(f"confusion_matrix_{model_name}.png")

    def plot_images_comparison(image1, image2, image3, image4):

        image1 = mpimg.imread('path_to_image1.jpg')
        image2 = mpimg.imread('path_to_image2.jpg')
        image3 = mpimg.imread('path_to_image3.jpg')
        image4 = mpimg.imread('path_to_image4.jpg')

        # Create a 1x4 subplot figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Adjust figsize for larger display if needed

        # Plot each image in a subplot
        axes[0].imshow(image1)
        axes[0].axis('off')  # Hide axis
        axes[0].set_title("Image 1")  # Optional: Set title

        axes[1].imshow(image2)
        axes[1].axis('off')
        axes[1].set_title("Image 2")

        axes[2].imshow(image3)
        axes[2].axis('off')
        axes[2].set_title("Image 3")

        axes[3].imshow(image4)
        axes[3].axis('off')
        axes[3].set_title("Image 4")

        # Display the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()
