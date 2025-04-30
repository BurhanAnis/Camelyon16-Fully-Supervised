import argparse
import os
import pickle
import matplotlib.pyplot as plt
from utils import evaluate_patch_predictions, plot_confusion_matrix

def main(args):
    # Load tile predictions
    with open(args.tile_predictions_file, "rb") as f:
        tile_predictions_dict = pickle.load(f)

    # Evaluate predictions
    results = evaluate_patch_predictions(
        tile_predictions_dict,
        args.annotations_dir,
        patch_size=args.patch_size,
        threshold=args.threshold
    )

    # Prepare output paths
    os.makedirs(args.results_dir, exist_ok=True)
    txt_output_path = os.path.join(args.results_dir, "evaluation_results_patch.txt")
    pkl_output_path = os.path.join(args.results_dir, "evaluation_results_patch.pkl")
    cm_output_path = os.path.join(args.results_dir, "confusion_matrix_patch.png")

    # Save evaluation results to text file
    with open(txt_output_path, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Sensitivity: {results['sensitivity']:.4f}\n")
        f.write(f"Specificity: {results['specificity']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")


    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    plot_confusion_matrix(cm)
    plt.savefig(cm_output_path)

    # Save results to file
    with open(pkl_output_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate patch-level tile predictions.")

    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save evaluation results and plots")
    parser.add_argument("--tile_predictions_file", type=str, required=True, help="Path to the tile predictions pickle file")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing ground truth annotations")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size used in tile prediction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to classify patch as positive")

    args = parser.parse_args()
    main(args)
