"""
t-SNE Visualization Script

Performs t-SNE dimensionality reduction on feature data and generates a visualization plot.
"""

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    # Set environment variable for CPU cores
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust based on your CPU cores

    # Ignore specific warnings
    import warnings
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

    # Define directories and step
    features_dir = 'gpt2-xl_mnli_09-26_08-58-59_r8_alpha16/features'
    output_dir = os.getcwd()
    step = 'final'

    # Load feature and label files
    features_file = os.path.join(features_dir, f'features_step_{step}.npy')
    labels_file = os.path.join(features_dir, f'labels_step_{step}.npy')

    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        print(f"Feature or label file missing: {features_file}, {labels_file}")
        exit()

    features = np.load(features_file)
    labels = np.load(labels_file)

    # Map numeric labels to text
    label_mapping = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

    # Perform t-SNE dimensionality reduction
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=1)
    features_2d = tsne.fit_transform(features)

    # Plot settings
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', alpha=0.5)

    # Add legend
    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(labels)
    class_names = [label_mapping[label] for label in unique_labels]
    legend = ax.legend(handles, class_names, fontsize=21, loc='upper left', bbox_to_anchor=(0.01, 0.99))

    # Save plot as PDF
    output_file = os.path.join(output_dir, f'tsne_step_{step}.pdf')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"t-SNE plot saved to {output_file}")


if __name__ == "__main__":
    main()
