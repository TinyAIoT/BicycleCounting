from pathlib import Path
from typing import Final

ROOT_DIR: Final[Path] = Path(__file__).parent.parent.resolve()

# Main source and output directories
CORE_DIR: Final[Path] = ROOT_DIR / "core"
DATA_DIR: Final[Path] = ROOT_DIR.parent / "data"
DATASETS_DIR: Final[Path] = ROOT_DIR.parent.parent / "model-training" / "datasets"
#MODELS_DIR: Final[Path] = ROOT_DIR.parent / "models"
MODELS_DIR: Final[Path] = ROOT_DIR.parent / "coco_detect" / "models"
UTILS_DIR: Final[Path] = ROOT_DIR / "utils"


CALIBRATION_IMAGE_DIR: Final[Path] = DATA_DIR / "calib_images_compressed"
ORIGINAL_IMAGE_DIR: Final[Path] = DATASETS_DIR / "combined_preprocessed" / "YOLO" / "images"
ORIGINAL_LABEL_DIR: Final[Path] = DATASETS_DIR / "combined_preprocessed" / "YOLO" / "labels"

GROUND_TRUTH_CSV_DIR: Final[Path] = DATA_DIR / "ground_truth_csvs"
BASE_MODEL_PRED_DIR: Final[Path] = DATA_DIR / "preds_base_model"
QUANTIZED_MODEL_PRED_DIR: Final[Path] = DATA_DIR / "preds_quantized_model"
ESP_MODEL_PRED_DIR: Final[Path] = DATA_DIR / "preds_esp_model"

BASE_MODEL_PT_PATH: Final[Path] = MODELS_DIR / "yolo11n.pt"
ONNX_MODEL_PATH: Final[Path] = MODELS_DIR / "yolo11n.onnx"
ESPDL_MODEL_PATH: Final[Path] = MODELS_DIR / "yolo11n.espdl"
