import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.path import Path

def parse_xml_annotations(xml_file):
    """Parses XML annotations to extract tumor regions."""
    if not os.path.exists(xml_file):
        print(f"Warning: Missing annotation file {xml_file}")
        return []
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    for annotation in root.findall(".//Annotation"):
        coords = []
        for coord in annotation.findall(".//Coordinate"):
            x = float(coord.attrib["X"])
            y = float(coord.attrib["Y"])
            coords.append((x, y))
        annotations.append(np.array(coords))
    
    return annotations

    def is_point_in_polygon(point, polygon):
    """Check if point is inside polygon using matplotlib Path."""
    path = Path(polygon)
    return path.contains_point(point)

    def is_patch_in_tumor_region(x, y, patch_size, annotations):
    """Check if a patch center is inside any tumor region."""
    # If there are no annotations, the slide is all negative
    if not annotations:
        return False
    
    # Use the center of the patch
    center_x = x + patch_size / 2
    center_y = y + patch_size / 2
    
    for polygon in annotations:
        if is_point_in_polygon((center_x, center_y), polygon):
            return True
    return False


def evaluate_patch_predictions(tile_predictions_dict, annotations_dir, patch_size=256, threshold=0.5):
    """
    Evaluate patch predictions against ground truth annotations.
    
    Args:
        tile_predictions_dict: Dictionary with slide names as keys and lists of (x, y, tumor_prob) as values
        annotations_dir: Directory containing XML annotation files
        patch_size: Size of each patch
        threshold: Threshold for classifying as tumor (probability >= threshold is tumor)
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_true = []
    y_pred = []
    
    for slide_name, predictions in tqdm(tile_predictions_dict.items(), desc="Evaluating Slides"):
        # Construct the XML file path - adjust this path format to match your structure
        # Assuming XML files have same name as slide but with .xml extension
        slide_id = os.path.splitext(slide_name)[0]
        xml_file = os.path.join(annotations_dir, f"{slide_id}.xml")
        
        if not os.path.exists(xml_file):
            print(f"Skipping {slide_name} - annotation file not found.")
            continue
        
        annotations = parse_xml_annotations(xml_file)
        
        for x, y, tumor_prob in predictions:
            # Get ground truth
            is_tumor = is_patch_in_tumor_region(x, y, patch_size, annotations)
            # Get prediction
            predicted_tumor = tumor_prob >= threshold
            
            y_true.append(1 if is_tumor else 0)
            y_pred.append(1 if predicted_tumor else 0)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Same as sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Create results dictionary
    results = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,  # Same as recall
        'specificity': specificity,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    return results

def plot_confusion_matrix(cm, classes=['Normal', 'Tumor'], title='Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt