"""
t-SNE Visualization Script
Performs t-SNE dimensionality reduction on the last hidden layer features from fine-tuning and generates a visualization plot.

Usage:
    Modify the `last_hidden_features_dir`, `output_dir`, and `step` variables below to configure the script.
"""

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Configuration variables
last_hidden_features_dir = '/path/to/last_hidden_features'  # Directory where the pre-saved last hidden layer features and labels are stored
output_dir = os.getcwd()                                   # Directory to save the output plot (default: current working directory)
step = 'final'                                             # Step name for loading feature and label files


def load_data(features_dir, step):
    """Load pre-saved last hidden layer features and corresponding labels."""
    features_file = os.path.join(features_dir, f'features_step_{step}.npy')
    labels_file = os.path.join(features_dir, f'labels_step_{step}.npy')

    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Feature or label file missing: {features_file}, {labels_file}")

    print("Loading pre-saved last hidden layer features and labels...")
    return np.load(features_file), np.load(labels_file)


def perform_tsne(features):
    """Perform t-SNE dimensionality reduction on the last hidden layer features."""
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=1)
    return tsne.fit_transform(features)


def plot_tsne(features_2d, labels, output_dir, step):
    """Generate and save the t-SNE visualization plot."""
    print("Generating t-SNE plot...")
    label_mapping = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', alpha=0.5)

    # Add legend
    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(labels)
    class_names = [label_mapping[label] for label in unique_labels]
    ax.legend(handles, class_names, fontsize=12, loc='upper left', bbox_to_anchor=(0.01, 0.99))

    # Save plot
    output_file = os.path.join(output_dir, f'tsne_step_{step}.pdf')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"t-SNE plot saved to {output_file}")


def main():
    """Main function to execute the t-SNE visualization pipeline."""
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust based on your CPU cores

    # Ignore specific warnings
    import warnings
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

    try:
        # Load pre-saved last hidden layer features and labels
        features, labels = load_data(last_hidden_features_dir, step)

        # Perform t-SNE dimensionality reduction
        features_2d = perform_tsne(features)

        # Generate and save t-SNE plot
        plot_tsne(features_2d, labels, output_dir, step)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
