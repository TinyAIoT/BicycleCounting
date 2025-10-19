from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from core.constants import TXT_ENCODING

# Arguments


class YoloTrainArgs(BaseModel):
    data: str = Field(..., description="Path to dataset YAML file")
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    device: Optional[str] = Field("cpu", description="Device to train on")
    val: Optional[bool] = Field(True, description="Enable validation")
    plots: Optional[bool] = Field(False, description="Save validation plots during training")

    class Config:
        extra = "allow"


class DataSplitArgs(BaseModel):
    weights: tuple[float, float, float] = Field(..., description="Ratios for dataset splitting (train, val, test)")
    annotated_only: bool = Field(..., description="Whether to split annotated images only")

    class Config:
        extra = "forbid"


class ValidationArgs(BaseModel):
    split: str = Field("test", description="Data split for validation")
    save_json: bool = Field(True, description="Whether to save validation results as JSON")
    save_txt: bool = Field(True, description="Whether to save detection results as TXT")
    task: str = Field("detect", description="YOLO task for validation")

    class Config:
        extra = "allow"
        exclude = ["data"]  # will be used from DataConfigSchema


class QuantizationArgs(BaseModel):
    calib_steps: int = Field(32, gt=0, description="Number of steps for calibration")
    num_bits: Literal[8, 16] = Field(8, description="Number of bits used for quantization")
    dispatching_override: Optional[dict[str, Any]] = Field(None, description="Override default dispatching settings")

    class Config:
        extra = "allow"


class QuantizationAwareTrainingArgs(BaseModel):
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    learning_rate: float = Field(3e-5, gt=0, description="Learning rate for training")
    device: Literal["cpu", "cuda"] = Field("cpu", description="Device used during training. Only cpu or cuda.")
    scheduling: Optional[Literal["linear"]] = Field(None, description="Learning Rate Scheduling method for training")
    scheduler_params: dict[str, Any] = Field({}, description="Learning Rate Scheduling parameters for training")

    class Config:
        extra = "allow"


# Configs


class BaseConfig(BaseModel):
    project_name: str = Field(..., description="Name of the model run")
    output_dir: str = Field(..., description="Directory to save model runs")
    model: str = Field(..., description="YOLO model name or path to weights")


class TrainConfig(BaseConfig):
    train_args: YoloTrainArgs = Field(..., description="Training arguments for Ultralytics YOLO class")
    data_split_args: Optional[DataSplitArgs] = Field(
        None, description="Data split arguments for Ultralytics YOLO class"
    )
    val_args: Optional[ValidationArgs] = Field(None, description="Validation arguments for Ultralytics YOLO class")

    class Config:
        extra = "forbid"

    @classmethod
    @field_validator("output_dir", mode="before")
    def ensure_output_dir(cls, v):
        if isinstance(v, str):
            v = Path(v)
        if not v.is_dir():
            raise ValueError(f"{v} is not a directory")
        if not v.exists():
            raise ValueError(f"Directory {v} does not exist")

    @model_validator(mode="after")
    def set_default_val_args(self) -> "TrainConfig":
        if self.val_args is None:
            # using default values
            self.val_args = ValidationArgs()
        return self


class DataConfig(BaseModel):
    path: str = Field(..., description="Absolute path to dataset directory containing images/ and labels/ directories")
    train: str = Field(..., description="Path to training directory or .txt file, relative to 'path' argument")
    val: str = Field(..., description="Path to validation directory or .txt file, relative to 'path' argument")
    test: Optional[str] = Field(..., description="Path to test directory or .txt file, relative to 'path' argument")
    nc: int = Field(..., description="Number of classes contained in the dataset")
    names: dict[int, str] = Field(..., description="Mapping from class ID to class name")


class QuantizationAwareTrainingConfig(BaseConfig):
    calib_dataset_path: str = Field(..., description="Path to calibration dataset directory.")
    dataset_yaml_file_path: str = Field(..., description="Path to calibration dataset YAML file. Used by Ultralytics.")
    onnx_model_path: str = Field(..., description="Path to ONNX model file")
    input_shape: Sequence[int] = Field([3, 640, 640], description="Model input shape. Defaults to Yolo11n input shape")
    training_args: QuantizationAwareTrainingArgs = Field(..., description="Arguments for quantization-aware training")
    quantization_args: QuantizationArgs = Field(..., description="Quantization arguments relevant for QAT")
    num_workers: int = Field(0, description="Number of workers used during calibration and training")
    split: Optional[Literal["test", "val"]] = Field(None, description="Data split used for validation")
    save_metrics: bool = Field(True, description="Save metrics from model evaluations during training.")

    class Config:
        extra = "forbid"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "QuantizationAwareTrainingConfig":
        with config_path.open("r", encoding=TXT_ENCODING) as config_file:
            config = yaml.safe_load(config_file)
        return cls(**config)

    @classmethod
    @field_validator("calib_dataset_path", mode="before")
    def ensure_calib_dataset_dir(cls, v):
        if isinstance(v, str):
            v = Path(v)
        if not v.is_dir():
            raise ValueError(f"{v} is not a directory")
        if not v.exists():
            raise ValueError(f"Directory {v} does not exist")

    @classmethod
    @field_validator("dataset_yaml_file_path", mode="before")
    def ensure_dataset_yaml_file(cls, v):
        if isinstance(v, str):
            v = Path(v)
        if not v.is_file():
            raise FileNotFoundError(f"{v} is not a file")
        if not v.exists():
            raise FileNotFoundError(f"File {v} does not exist")
