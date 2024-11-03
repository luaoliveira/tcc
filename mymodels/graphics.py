import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix




import numpy as np
import plotly.graph_objects as go

def plot_confusion_matrix(tp, tn, fp, fn, model_name):
    """
    Plots a confusion matrix using counts of true positives, true negatives,
    false positives, and false negatives.
    
    Parameters:
    - tp: True Positives
    - tn: True Negatives
    - fp: False Positives
    - fn: False Negatives
    - model_name: Name of the model to use in the plot title
    """
    # Arrange the counts into a 2x2 confusion matrix
    conf_matrix = np.array([[tn, fp],  # First row: True Negatives, False Positives
                            [fn, tp]]) # Second row: False Negatives, True Positives

    # Normalize the confusion matrix by row sums, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)  # Replace NaN with zero for rows with sum=0

    # Define labels for the binary classes (0 = Background, 1 = Foreground)
    labels = ['Background', 'Foreground']
    
    # Create the heatmap using Plotly with normalized values
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix_normalized,
        x=labels,  # x-axis labels (Predicted)
        y=labels,  # y-axis labels (True)
        colorscale="Blues",
        hoverongaps=False,
        zmin=0,
        zmax=1,
        showscale=False
    ))

    # Add annotations for percentage values in each cell with dynamic font color
    for i in range(2):
        for j in range(2):
            percentage = conf_matrix_normalized[i, j] * 100  # Convert to percentage
            font_color = "white" if conf_matrix_normalized[i, j] > 0.5 else "black"  # Dynamic font color
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{percentage:.2f}%",  # Format as a percentage
                showarrow=False,
                font=dict(color=font_color)
            )

    # Add titles and display options
    fig.update_layout(
        title={
            'text': f"Normalized Confusion Matrix - {model_name}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )

    # Save the plot as an image
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

import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_false_negatives_positives(ground_truth_path, predicted_mask_path):
    """
    Reads two binary images (ground truth and predicted mask), applies a threshold to ensure binary values,
    computes false positives and false negatives, and plots them with different colors.

    Parameters:
    - ground_truth_path: Path to the ground truth binary image.
    - predicted_mask_path: Path to the predicted binary mask image.
    """
    # Read the images in grayscale
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply threshold to ensure binary values (0 or 1)
    _, ground_truth = cv2.threshold(ground_truth, 30, 1, cv2.THRESH_BINARY)
    _, predicted_mask = cv2.threshold(predicted_mask, 30, 1, cv2.THRESH_BINARY)

    # Calculate false positives and false negatives
    false_positives = np.logical_and(predicted_mask == 1, ground_truth == 0)  # Predicted 1, Actual 0
    false_negatives = np.logical_and(predicted_mask == 0, ground_truth == 1)  # Predicted 0, Actual 1

    # Create an empty color map for visualization
    display_image = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)

    # Color false positives red and false negatives blue
    display_image[false_positives] = [255, 0, 0]  # Red for false positives
    display_image[false_negatives] = [0, 0, 255]  # Blue for false negatives

    # Plot the results
    plt.figure(figsize=(8, 8))
    plt.imshow(display_image)
    plt.axis('off')

    # Create a legend for the colors
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='False Positive (Pred=1, GT=0)',
                           markerfacecolor='red', markersize=10)
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='False Negative (Pred=0, GT=1)',
                            markerfacecolor='blue', markersize=10)

    plt.legend(handles=[red_patch, blue_patch], loc="upper right")
    plt.title("False Positives and False Negatives")
    # plt.show()
    plt.savefig("false_positives_negatives.png", bbox_inches='tight', dpi=300)


