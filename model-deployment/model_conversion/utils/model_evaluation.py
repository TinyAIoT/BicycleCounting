import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from pandas.errors import EmptyDataError
from ultralytics import YOLO
from ultralytics.engine.results import Results
from tqdm import tqdm

from model_conversion.core.paths import (
    CALIBRATION_IMAGE_DIR, BASE_MODEL_PRED_DIR, GROUND_TRUTH_CSV_DIR,
    QUANTIZED_MODEL_PRED_DIR
)
from model_conversion.core.constants import (
    CONF_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS, CLASS_NAMES, MODEL_MEAN,
    MODEL_STD, MODEL_INPUT_SHAPE
)


class YoloDetector:
    """Handles object detection using the original YOLOv8 .pt model."""

    def __init__(
            self,
            model_path: str,
            conf_threshold: float = CONF_THRESHOLD,
            iou_threshold: float = IOU_THRESHOLD,
            max_detections: int = MAX_DETECTIONS,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        print("YOLO model loaded successfully.")

    def predict_on_image(self, image_path: Path) -> Results:
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )
        return results[0]

    @staticmethod
    def save_results(results: Results, output_dir: Path, original_image_path: Path):
        output_csv_file = output_dir / f"{original_image_path.stem}.csv"
        # Standardized flat CSV format
        pred_df = pd.DataFrame(results.boxes.data.cpu().numpy(), columns=[
            'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'
        ])
        pred_df['class_name'] = [results.names[int(cls_id)] for cls_id in pred_df['class_id']]
        pred_df.to_csv(output_csv_file, index=False)

    def process_directory(self, image_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing images from: {image_dir}")
        print(f"Saving predictions to: {output_dir}")
        image_paths = sorted(list(image_dir.glob("*.jpg")))
        if not image_paths:
            print(f"No .jpg images found in {image_dir}")
            return
        for img_path in tqdm(image_paths, desc="Detecting objects"):
            results = self.predict_on_image(img_path)
            self.save_results(results, output_dir, img_path)
        print("\nProcessing complete.")


class ESPEvaluator:
    """
    Handles preprocessing, postprocessing, and performance evaluation for
    both the original and quantized models.
    """

    def __init__(
            self,
            image_dir: Path = CALIBRATION_IMAGE_DIR,
            gt_dir: Path = GROUND_TRUTH_CSV_DIR,
            base_pred_dir: Path = BASE_MODEL_PRED_DIR,
            quantized_pred_dir: Path = QUANTIZED_MODEL_PRED_DIR,
            class_names: Dict[int, str] = CLASS_NAMES,
            model_mean: List[int] = MODEL_MEAN,
            model_std: List[int] = MODEL_STD,
            input_shape: Tuple[int, int] = MODEL_INPUT_SHAPE,
            conf_threshold: float = CONF_THRESHOLD,
            iou_threshold: float = IOU_THRESHOLD,
            max_detections: int = MAX_DETECTIONS,
    ):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.base_pred_dir = base_pred_dir
        self.quantized_pred_dir = quantized_pred_dir
        self.class_names = class_names
        self.model_mean = model_mean
        self.model_std = model_std
        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def preprocess_for_esp_dl(self, image_path, model_input_shape, mean, std):
        img_bgr = cv2.imread(image_path)
        assert img_bgr is not None, f"Image not found at {image_path}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        target_h, target_w = model_input_shape
        resized_img = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        img_tensor = torch.from_numpy(resized_img).float()
        mean_tensor = torch.tensor(mean, dtype=torch.float32).reshape(1, 1, 3)
        std_tensor = torch.tensor(std, dtype=torch.float32).reshape(1, 1, 3)
        normalized_tensor = (img_tensor - mean_tensor) / std_tensor
        input_tensor = normalized_tensor.permute(2, 0, 1).unsqueeze(0)
        return input_tensor

    def postprocess_for_esp_dl(self, outputs, conf_threshold, iou_threshold, max_detections):
        strides = [8, 16, 32]
        reg_max = 16
        bins = torch.arange(reg_max, device=outputs[0].device, dtype=torch.float32)
        all_boxes, all_scores, all_class_ids = [], [], []
        box_preds, cls_preds = [outputs[0], outputs[2], outputs[4]], [outputs[1], outputs[3], outputs[5]]

        for i, stride in enumerate(strides):
            box_pred, cls_pred = box_preds[i], cls_preds[i]
            height, width = cls_pred.shape[2], cls_pred.shape[3]
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(1, -1, cls_pred.shape[1])[0]
            box_pred = box_pred.permute(0, 2, 3, 1).reshape(1, -1, 4 * reg_max)[0]
            scores, class_ids = torch.sigmoid(cls_pred).max(1)
            confident_mask = scores > conf_threshold
            if not confident_mask.any(): continue

            grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            grid_coords = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(box_pred.device)
            confident_coords = (grid_coords[confident_mask] + 0.5) * stride
            box_reg_dist = torch.softmax(box_pred[confident_mask].reshape(-1, reg_max), dim=1).matmul(bins).reshape(-1,
                                                                                                                    4) * stride
            d_left, d_top, d_right, d_bottom = box_reg_dist.chunk(4, dim=1)

            # Calculate the top-left (x1, y1) and bottom-right (x2, y2) coordinates as tensors
            top_left = confident_coords - torch.cat([d_left, d_top], dim=1)
            bottom_right = confident_coords + torch.cat([d_right, d_bottom], dim=1)

            # Concatenate them to create the final [N, 4] bounding box tensor
            decoded_boxes = torch.cat([top_left, bottom_right], dim=1)

            all_boxes.append(decoded_boxes)
            all_scores.append(scores[confident_mask])
            all_class_ids.append(class_ids[confident_mask])

        if not all_boxes: return []
        final_boxes, final_scores, final_class_ids = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(
            all_class_ids)
        nms_indices = torchvision.ops.nms(final_boxes, final_scores, iou_threshold)
        nms_indices = nms_indices[:max_detections]

        return [
            (final_class_ids[i].item(), final_scores[i].item(), *final_boxes[i].cpu().numpy().astype(int))
            for i in nms_indices
        ]

    def load_ground_truth_from_csv(self, csv_path):
        if not os.path.exists(csv_path): return torch.tensor([]), torch.tensor([])
        try:
            df = pd.read_csv(csv_path)
            if df.empty: return torch.tensor([]), torch.tensor([])
        except EmptyDataError:
            return torch.tensor([]), torch.tensor([])
        gt_boxes = df[['x1', 'y1', 'x2', 'y2']].values
        gt_class_ids = df['class_id'].values
        return torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(gt_class_ids, dtype=torch.int64)

    def calculate_iou(self, box1, box2):
        x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
        x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
        return (intersection / union).item()

    def calculate_ap_for_class_across_images(self, predictions, ground_truths_by_image, total_gt_count,
                                             iou_threshold=0.5):
        if total_gt_count == 0:
            fp_count = len(predictions)
            return (1.0 if not predictions else 0.0), [], 0, fp_count, 0

        predictions.sort(key=lambda x: x[2], reverse=True)
        tp, fp = np.zeros(len(predictions)), np.zeros(len(predictions))
        gt_used_map = {img_name: [False] * len(boxes) for img_name, boxes in ground_truths_by_image.items()}
        true_positive_ious = []

        for i, (img_name, pred_box, score) in enumerate(predictions):
            gt_boxes = ground_truths_by_image.get(img_name, [])
            best_iou, best_gt_idx = 0, -1
            for j, gt_box in enumerate(gt_boxes):
                iou = self.calculate_iou(torch.tensor(pred_box), torch.tensor(gt_box))
                if iou > best_iou: best_iou, best_gt_idx = iou, j
            if best_iou >= iou_threshold and not gt_used_map[img_name][best_gt_idx]:
                tp[i], gt_used_map[img_name][best_gt_idx] = 1, True
                true_positive_ious.append(best_iou)
            else:
                fp[i] = 1

        tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cumsum / total_gt_count
        precisions = np.divide(tp_cumsum, (tp_cumsum + fp_cumsum), out=np.zeros_like(tp_cumsum, dtype=float),
                               where=(tp_cumsum + fp_cumsum) != 0)
        precisions, recalls = np.concatenate(([0.], precisions, [0.])), np.concatenate(([0.], recalls, [1.]))
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        ap = sum((recalls[i + 1] - recalls[i]) * precisions[i + 1] for i in range(len(recalls) - 1))

        tp_count, fp_count, fn_count = int(np.sum(tp)), int(np.sum(fp)), total_gt_count - int(np.sum(tp))
        return ap, true_positive_ious, tp_count, fp_count, fn_count

    def save_predictions_to_csv(self, detections_tensor, class_names_map, output_csv_path):
        columns = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name']
        if detections_tensor is None or detections_tensor.numel() == 0:
            pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
            return
        rows = []
        for det in detections_tensor:
            x1, y1, x2, y2, score, class_id_tensor = det
            class_id = int(class_id_tensor.item())
            class_name = class_names_map.get(class_id, f"class_{class_id}")
            rows.append({'x1': x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item(),
                         'confidence': score.item(), 'class_id': class_id, 'class_name': class_name})
        pd.DataFrame(rows, columns=columns).to_csv(output_csv_path, index=False)

    def load_predictions_from_csv(self, csv_path):
        if not os.path.exists(csv_path): return torch.tensor([]), torch.tensor([]), torch.tensor([])
        try:
            df = pd.read_csv(csv_path)
            if df.empty: return torch.tensor([]), torch.tensor([]), torch.tensor([])
        except EmptyDataError:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        pred_boxes = df[['x1', 'y1', 'x2', 'y2']].values
        pred_class_ids = df['class_id'].values
        pred_scores = df['confidence'].values
        return torch.tensor(pred_boxes), torch.tensor(pred_class_ids), torch.tensor(pred_scores)

    def calculate_metrics_from_collected_data(self, all_predictions, all_ground_truths):
        results = defaultdict(dict)
        # Initialize dictionaries for P, R, and counts
        results["precision_per_class"] = {}
        results["recall_per_class"] = {}
        results["tps_per_class"] = {}
        results["fps_per_class"] = {}
        results["fns_per_class"] = {}
        results["ap_per_class"] = {}
        results["avg_iou_per_class"] = {}

        all_class_ids = sorted(list(set(all_predictions.keys()) | set(all_ground_truths.keys())))

        for class_id in all_class_ids:
            class_preds = all_predictions.get(class_id, [])
            class_gts_raw = all_ground_truths.get(class_id, [])
            class_gts_by_image = defaultdict(list)
            for img_name, box_list in class_gts_raw:
                class_gts_by_image[img_name].append(box_list)

            total_gt_count = len(class_gts_raw)
            ap, tp_ious, tp, fp, fn = self.calculate_ap_for_class_across_images(
                class_preds, class_gts_by_image, total_gt_count, iou_threshold=0.5
            )

            # Store the raw counts
            results["tps_per_class"][class_id] = tp
            results["fps_per_class"][class_id] = fp
            results["fns_per_class"][class_id] = fn
            results["ap_per_class"][class_id] = ap
            results["avg_iou_per_class"][class_id] = np.mean(tp_ious) if tp_ious else 0.0

            # Calculate and store Precision and Recall
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-9
            precision = tp / (tp + fp + epsilon)
            recall = tp / (total_gt_count + epsilon)  # total_gt_count is TP + FN

            results["precision_per_class"][class_id] = precision
            results["recall_per_class"][class_id] = recall

        valid_aps = [v for v in results["ap_per_class"].values() if not np.isnan(v)]
        results["mAP"] = np.mean(valid_aps) if valid_aps else 0.0

        # Calculate macro-average Precision and Recall
        valid_precisions = [v for v in results["precision_per_class"].values()]
        valid_recalls = [v for v in results["recall_per_class"].values()]
        results["macro_precision"] = np.mean(valid_precisions) if valid_precisions else 0.0
        results["macro_recall"] = np.mean(valid_recalls) if valid_recalls else 0.0

        return results

    def evaluate_csv_predictions(self, image_paths, gt_dir, prediction_dir):
        all_predictions, all_ground_truths = defaultdict(list), defaultdict(list)
        print(f"\nEvaluating pre-computed CSVs from '{prediction_dir}'...")
        for image_path in image_paths:
            image_base_name = Path(image_path).stem
            gt_boxes, gt_classes = self.load_ground_truth_from_csv(gt_dir / f"{image_base_name}.csv")
            for box, cls_id in zip(gt_boxes, gt_classes):
                all_ground_truths[cls_id.item()].append([image_base_name, box.tolist()])

            pred_boxes, pred_classes, pred_scores = self.load_predictions_from_csv(
                prediction_dir / f"{image_base_name}.csv")
            for box, cls_id, score in zip(pred_boxes, pred_classes, pred_scores):
                all_predictions[cls_id.item()].append([image_base_name, box.tolist(), score.item()])

        return self.calculate_metrics_from_collected_data(all_predictions, all_ground_truths)


class BoundingBoxVisualizer:

    def __init__(
            self,
            class_names: Dict[int, str],
            colors: Dict[int, Tuple[int, int, int]],
            font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
            font_scale: float = 0.5,
            box_thickness: int = 2,
            text_thickness: int = 2,
    ):

        self.class_names = class_names
        self.colors = colors
        self.font_face = font_face
        self.font_scale = font_scale
        self.box_thickness = box_thickness
        self.text_thickness = text_thickness

    def _draw_single_box(self, image: np.ndarray, prediction: Tuple):

        class_id, score, x1, y1, x2, y2 = prediction

        # Ensure coordinates are integers for drawing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get the color and name for the class, with fallbacks for unknown classes
        color = self.colors.get(class_id, (0, 255, 0))  # Default to green
        name = self.class_names.get(class_id, f"Class {class_id}")

        label = f"{name}: {score:.2f}"

        # Draw the bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)

        # Draw the label text above the box
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            self.font_face,
            self.font_scale,
            color,
            self.text_thickness,
        )

    def draw_boxes_on_image(
            self,
            image_path: Path,
            predictions: List[Tuple],
            output_path: Path,
    ) -> np.ndarray:

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image from: {image_path}")

        annotated_image = image.copy()

        for pred in predictions:
            self._draw_single_box(annotated_image, pred)

        cv2.imwrite(str(output_path), annotated_image)
        print(f"Annotated image saved to: {output_path}")
        return annotated_image
