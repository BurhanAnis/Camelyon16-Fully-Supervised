import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def main(args):
    # Load data
    data = pd.read_csv(args.reference_csv, header=None)
    preds = load_data(args.predictions_file)

    # Binarize predictions
    ground_truth = data[1].map({'Tumor': 1, 'Normal': 0})
    preds_binary = [1 if v > args.threshold else 0 for v in preds.values()]

    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, preds_binary)

    # Save confusion matrix plot
    os.makedirs(args.results_dir, exist_ok=True)
    cm_output_path = os.path.join(args.results_dir, "slide_confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Tumor (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(cm_output_path)
    plt.show()

    # Compute metrics
    acc = accuracy_score(ground_truth, preds_binary)
    prec = precision_score(ground_truth, preds_binary)
    rec = recall_score(ground_truth, preds_binary)
    f1 = f1_score(ground_truth, preds_binary)

    # Save metrics to text file
    txt_output_path = os.path.join(args.results_dir, "slide_level_metrics.txt")
    with open(txt_output_path, "w") as f:
        f.write("Slide-Level Evaluation Metrics:\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall (Sensitivity): {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate slide-level predictions and save results.")
    parser.add_argument("--reference_csv", type=str, required=True, help="Path to the reference CSV file with ground truth labels.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to the slide-level predictions pickle file.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize predictions.")

    args = parser.parse_args()
    main(args)


