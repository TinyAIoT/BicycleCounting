import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

from model_conversion.utils.model_evaluation import ESPEvaluator
from model_conversion.core.paths import (
    CALIBRATION_IMAGE_DIR, GROUND_TRUTH_CSV_DIR,
    QUANTIZED_MODEL_PRED_DIR, BASE_MODEL_PRED_DIR,ESP_MODEL_PRED_DIR
)
from model_conversion.core.constants import CLASS_NAMES

PREDICTION_DIR = QUANTIZED_MODEL_PRED_DIR
# PREDICTION_DIR = BASE_MODEL_PRED_DIR
# PREDICTION_DIR = ESP_MODEL_PRED_DIR
VISUALIZATION_OUTPUT_DIR = Path("./data/evaluation/visuals_quantized")
# VISUALIZATION_OUTPUT_DIR = Path("./data/evaluation/visuals_base")
# VISUALIZATION_OUTPUT_DIR = Path("./evaluation_visuals_esp")
IOU_THRESHOLD = 0.50 # To visualize AP@50

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

GT_COLOR = (0, 255, 0)  # Green
TP_COLOR = (255, 0, 0)  # Blue
FP_COLOR = (0, 0, 255)  # Red
FN_COLOR = (0, 255, 255)  # Yellow
TEXT_COLOR = (0, 0, 0)  # Black


def match_predictions_to_gt(predictions, gt_boxes, gt_classes):
    """
    Matches predictions to ground truth boxes for a single image.

    Returns:
        A list of dictionaries, where each dict is a prediction with its status (TP/FP).
        A list of booleans indicating which GT boxes were matched.
    """
    if len(predictions) == 0:
        return [], [False] * len(gt_boxes)
    if len(gt_boxes) == 0:
        for p in predictions: p['status'] = 'FP'
        return predictions, []

    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    evaluator = ESPEvaluator()

    for pred in predictions:
        pred['status'] = 'FP'
        best_iou = 0
        best_gt_idx = -1

        for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if int(gt_cls) == int(pred['class_id']):
                iou = evaluator.calculate_iou(torch.tensor(pred['box']), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        if best_gt_idx != -1 and best_iou >= IOU_THRESHOLD and not gt_matched[best_gt_idx]:
            pred['status'] = 'TP'
            pred['iou'] = best_iou
            gt_matched[best_gt_idx] = True

    return predictions, gt_matched


def draw_results_on_image(image_path, predictions, gt_boxes, gt_classes, gt_matched):
    """Draws all boxes on the image and saves it."""
    image = cv2.imread(str(image_path))

    # Draw all Ground Truth boxes first (so they are in the background)
    for i, gt_box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, gt_box)
        color = FN_COLOR if not gt_matched[i] else GT_COLOR  # Unmatched GTs (FNs) are yellow
        label = "FN" if not gt_matched[i] else "GT"  # Unmatched GTs (FNs) are yellow
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        class_name = CLASS_NAMES.get(int(gt_classes[i]), "Unknown")
        cv2.putText(image, f"{label}: {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw all Prediction boxes
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['box'])
        class_name = CLASS_NAMES.get(int(pred['class_id']), "Unknown")
        conf = pred['confidence']

        if pred['status'] == 'TP':
            color = TP_COLOR
            label = f"TP: {class_name} ({conf:.2f}, IoU:{pred['iou']:.2f})"
        # if pred['status'] == 'FN':
        #     color = FN_COLOR
        #     label = f"FN: {class_name}"
        else:
            color = FP_COLOR
            label = f"FP: {class_name} ({conf:.2f})"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def filter_data_by_class(target_class_id, boxes, classes, scores=None):
    """
    Filters tensor data to keep only entries for a specific class ID.
    This now uses efficient PyTorch tensor indexing.
    """
    if target_class_id is None:
        return boxes, classes, scores

    # Create a boolean mask where the condition is true
    mask = (classes == target_class_id)

    # Apply the mask to all tensors
    filtered_boxes = boxes[mask]
    filtered_classes = classes[mask]

    if scores is not None:
        filtered_scores = scores[mask]
        return filtered_boxes, filtered_classes, filtered_scores
    else:
        return filtered_boxes, filtered_classes


def main():
    """Main function to generate visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation visualizations. Can filter by a specific class."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help=f"Optional: Visualize for a single class. Choices are: {list(NAME_TO_ID.keys())}"
    )
    args = parser.parse_args()

    target_class_id = None
    output_dir = VISUALIZATION_OUTPUT_DIR

    if args.class_name:
        if args.class_name not in NAME_TO_ID:
            print(f"Error: Invalid class name '{args.class_name}'.")
            print(f"Valid options are: {list(NAME_TO_ID.keys())}")
            sys.exit(1)
        target_class_id = NAME_TO_ID[args.class_name]
        output_dir = output_dir / args.class_name
        print(f"\n--- Filtering for class: '{args.class_name}' (ID: {target_class_id}) ---\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ESPEvaluator()

    image_paths = sorted(list(CALIBRATION_IMAGE_DIR.glob("*.jpg")))
    error_summary = []

    print(f"Generating visualizations for {len(image_paths)} images...")
    for image_path in tqdm(image_paths):
        image_base_name = image_path.stem

        # Load ALL ground truth and predictions for the image
        gt_csv_path = GROUND_TRUTH_CSV_DIR / f"{image_base_name}.csv"
        all_gt_boxes, all_gt_classes = evaluator.load_ground_truth_from_csv(gt_csv_path)

        pred_csv_path = PREDICTION_DIR / f"{image_base_name}.csv"
        all_pred_boxes, all_pred_classes, all_pred_scores = evaluator.load_predictions_from_csv(pred_csv_path)

        # By default, we use all data.
        gt_boxes_to_show = all_gt_boxes
        gt_classes_to_show = all_gt_classes
        pred_boxes_to_eval = all_pred_boxes
        pred_classes_to_eval = all_pred_classes
        pred_scores_to_eval = all_pred_scores

        # If a class is specified, filter ALL data that will be processed and drawn.
        if target_class_id is not None:
            gt_boxes_to_show, gt_classes_to_show = filter_data_by_class(
                target_class_id, all_gt_boxes, all_gt_classes
            )
            pred_boxes_to_eval, pred_classes_to_eval, pred_scores_to_eval = filter_data_by_class(
                target_class_id, all_pred_boxes, all_pred_classes, all_pred_scores
            )

        # Format predictions into a list of dicts for easier processing
        predictions_list = [
            {'box': box.tolist(), 'class_id': int(cls), 'confidence': score.item()}
            for box, cls, score in zip(pred_boxes_to_eval, pred_classes_to_eval, pred_scores_to_eval)
        ]

        # Match predictions to GT to find TPs and FPs
        # This now only operates on the filtered data
        matched_preds, gt_matched_flags = match_predictions_to_gt(
            predictions_list, gt_boxes_to_show, gt_classes_to_show
        )

        # Count errors for this image
        fp_count = sum(1 for p in matched_preds if p['status'] == 'FP')
        fn_count = sum(1 for matched in gt_matched_flags if not matched)
        if fp_count > 0 or fn_count > 0:
            error_summary.append({
                'image': image_path.name,
                'fp_count': fp_count,
                'fn_count': fn_count
            })

        annotated_image = draw_results_on_image(
            image_path, matched_preds, gt_boxes_to_show, gt_classes_to_show, gt_matched_flags
        )

        # Save the output image
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_image)

    print("\nVisualizations saved to:", output_dir)

    print("\n--- Error Analysis Summary ---")
    if args.class_name:
        print(f"--- (Filtered for class: '{args.class_name}') ---")

    # Sort by most False Positives
    error_summary.sort(key=lambda x: x['fp_count'], reverse=True)
    print("\nTop 10 Images with Most False Positives (FPs):")
    if not error_summary:
        print("  - No errors found.")
    for item in error_summary[:10]:
        print(f"  - {item['image']}: {item['fp_count']} FPs, {item['fn_count']} FNs")

    # Sort by most False Negatives
    error_summary.sort(key=lambda x: x['fn_count'], reverse=True)
    print("\nTop 10 Images with Most False Negatives (FNs):")
    if not error_summary:
        print("  - No errors found.")
    for item in error_summary[:10]:
        print(f"  - {item['image']}: {item['fn_count']} FNs, {item['fp_count']} FPs")


if __name__ == "__main__":
    main()