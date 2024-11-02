import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix




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
        zmax=1
    ))

    # Add annotations for percentage values in each cell
    for i in range(2):
        for j in range(2):
            percentage = conf_matrix_normalized[i, j] * 100  # Convert to percentage
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{percentage:.2f}%",  # Format as a percentage
                showarrow=False,
                font=dict(color="black")
            )

    # Add titles and display options
    fig.update_layout(
        title=f"Normalized Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    # Display the plot
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
