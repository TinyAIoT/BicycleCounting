import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

# isort: off
import onnxruntime as ort
import ppq.lib as PFL
import torch
import yaml
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph, TrainableGraph
from ppq.parser import NativeExporter
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
import wandb
# isort: on

from model_training.core.constants import TXT_ENCODING, WANDB_PROJECT
from model_training.core.schemas import (
    DataConfig,
    QuantizationAwareTrainingArgs,
    QuantizationAwareTrainingConfig,
)
from model_training.utils.datasets import CalibrationDataset, TrainDataset
from model_training.utils.quantization import QuantizationSetup
from model_training.utils.validators import QuantDetectionValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuantizationAwareTrainer:
    """Trainer class for quantization-aware training of YOLO models"""

    def __init__(
        self,
        ppq_graph: BaseGraph,
        yolo_model: str | Path,
        onnx_model_path: Path,
        dataset_yaml_path: Path,
        training_arguments: QuantizationAwareTrainingArgs,
        num_bits: Literal[8, 16],
    ) -> None:
        """
        Initialize QAT pipeline
        :param ppq_graph: PPQ BaseGraph for quantized model
        :param yolo_model: Path to YOLO model
        :param onnx_model_path: Path to original ONNX model
        :param dataset_yaml_path: Path to dataset YAML file. Used by Ultralytics.
        :param training_arguments: Training arguments.
        :param num_bits: Precision of quantized model
        """
        self.ppq_graph = ppq_graph
        self.yolo_model = yolo_model if isinstance(yolo_model, Path) else Path(yolo_model)
        self.onnx_model_path = onnx_model_path
        self.dataset_yaml_path = dataset_yaml_path
        self.num_bits = num_bits
        self.epochs = training_arguments.epochs
        self.learning_rate = training_arguments.learning_rate
        self.device = training_arguments.device
        self.scheduling = training_arguments.scheduling
        self.scheduler_params = training_arguments.scheduler_params

        # PPQ graphs and native model files
        self._latest_native_model: Optional[Path] = None
        self._latest_espdl_model: Optional[Path] = None

        # training state
        self._curr_epoch = 0
        self._curr_step = 0
        self._best_pr = 0.0
        self._best_metrics: list[dict[str, Any]] = []
        self._best_epoch = 0

        # init QAT components
        self._executor = TorchExecutor(graph=self.ppq_graph, device=self.device)
        self._training_graph = TrainableGraph(self.ppq_graph)
        self._loss_fn = torch.nn.MSELoss()
        self._lr_scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None

        # set up optimizer and gradients for trainable parameters
        self._optimizer = self._get_optimizer()
        self._enable_gradients()

    @property
    def current_epoch(self) -> int:
        """Get current epoch number."""
        return self._curr_epoch

    @property
    def current_step(self) -> int:
        """Get current step number."""
        return self._curr_step

    @property
    def best_precision_recall(self) -> float:
        """Get best metric achieved so far."""
        return self._best_pr

    @property
    def best_metrics(self) -> list[dict]:
        """Get best metric achieved so far."""
        return self._best_metrics

    @property
    def latest_native_model(self) -> Optional[Path]:
        return self._latest_native_model

    @property
    def latest_espdl_model(self) -> Optional[Path]:
        return self._latest_espdl_model

    def _get_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(
            params=[{"params": self._training_graph.parameters()}],
            lr=self.learning_rate,
        )
        if self.scheduling:
            if not self.scheduler_params:
                logger.info("No learning rate scheduling parameters provided, using default scheduling.")
            match self.scheduling:
                case "linear":
                    self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, **self.scheduler_params)
                case _:
                    raise NotImplementedError(f"No learning rate scheduling defined for {self.scheduling}")
        return optimizer

    def _enable_gradients(self) -> None:
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True

    def train_epoch(self, train_dataloader: DataLoader) -> float:
        """Train for one epoch and return average epoch loss"""
        epoch_loss = 0.0
        num_batches = len(train_dataloader)

        if num_batches == 0:
            raise IOError("No training data found.")

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {self._curr_epoch}", total=num_batches if num_batches > 0 else None
        )

        for batch_idx, batch in enumerate(progress_bar):
            data = batch.to(self.device)
            _, loss = self._training_step(data)
            epoch_loss += loss

            # Update progress bar
            if num_batches > 0:
                progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
        self._curr_epoch += 1

        logger.info(f"Epoch {self._curr_epoch - 1} completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def _training_step(self, data: torch.Tensor) -> tuple[list[Tensor], Any]:
        """Performs one training step one a given batch"""
        # Forward pass through quantized model
        quantized_predictions = self._executor.forward_with_gradient(data)

        # Forward pass through original FP32 model
        fp32_predictions = self._get_fp32_predictions(data)

        # Compute loss between quantized and FP32 predictions
        total_loss = 0.0
        for i, (quant_pred, fp32_pred) in enumerate(zip(quantized_predictions, fp32_predictions)):
            fp32_tensor = torch.tensor(fp32_pred, device=self.device, dtype=torch.float32)
            loss = self._loss_fn(quant_pred, fp32_tensor)
            total_loss += loss

        # Backward pass
        total_loss.backward()  # type: ignore

        # Optimizer step
        self._optimizer.step()
        self._training_graph.zero_grad()

        # Update learning rate if scheduler is available
        if self._lr_scheduler:
            self._lr_scheduler.step()

        self._curr_step += 1
        return quantized_predictions, total_loss.item()  # type: ignore

    def _get_fp32_predictions(self, data: torch.Tensor) -> Sequence[Any]:
        """Get predictions from original FP32 ONNX model.

        :param data: Input tensor
        returns: List of FP32 model predictions
        """
        try:
            session = ort.InferenceSession(self.onnx_model_path.as_posix())
            input_name = session.get_inputs()[0].name
            numpy_data = data.cpu().numpy()
            return session.run(None, {input_name: numpy_data})
        except Exception as e:
            raise RuntimeError(f"Failed to run FP32 inference: {e}") from e

    def evaluate(self, save_metrics: bool, file_path: Optional[Path] = None):
        """
        Do Evaluation process on given dataloader.

        :param save_metrics: Whether to save metrics during evaluation in CSV format.
        :param file_path: Path to store evaluation results.
        """

        model = YOLO(self.yolo_model)
        model.to(self.device)
        results = model.val(
            data=self.dataset_yaml_path.as_posix(),
            # TODO: get from training arguments
            imgsz=640,
            device=self.device,
            # TODO: fix QuantDetectionValidator()
            validator=QuantDetectionValidator(
                args={
                    "onnx_model_path": self.onnx_model_path,
                    "native_model_path": self.latest_native_model,
                    "num_bits": self.num_bits,
                }
            ),
            split="val",
            verbose=True,
        )
        if save_metrics and results and file_path:
            csv_metrics = results.to_csv()
            with file_path.open("w", encoding=TXT_ENCODING) as f:
                f.write(csv_metrics)

        return results

    def save_model(self, espdl_path: Path, native_path: Path) -> None:
        """
        Saves intermediate espdl and native models during training.
        :param espdl_path: Path to espdl model file, including file name.
        :param native_path: Path to native model file, including file name.
        """

        match self.num_bits:
            case 8:
                espdl_exporter = PFL.Exporter(platform=TargetPlatform.ESPDL_INT8)
            case 16:
                espdl_exporter = PFL.Exporter(platform=TargetPlatform.ESPDL_INT16)
            case _:
                raise IOError(f"Invalid number of bits: {self.num_bits}. Only 8 or 16 are supported for quantization.")
        espdl_exporter.export(espdl_path.as_posix(), self.ppq_graph)
        self._latest_espdl_model = espdl_path

        native_exporter = NativeExporter()
        native_exporter.export(native_path.as_posix(), self.ppq_graph)
        self._latest_native_model = native_path

    def update_metrics(self, scores: DetMetrics) -> bool:
        """
        Keeps track of best metrics from model runs
        :param scores: Detection Metrics from Ultralytics
        """
        if scores:
            # if Precision-Recall is the highest, keep track of best model results
            if scores.curves_results[0] > self.best_precision_recall:
                self._best_metrics = json.loads(scores.to_json())
                self._best_epoch = self.current_epoch
                return True
        return False


class QuantizationAwareTrainingPipeline:
    """Implements the full pipline for YOLO quantization-aware training."""

    def __init__(self, config: QuantizationAwareTrainingConfig) -> None:
        """Initialize pipeline"""
        self.config = config

        self.model_path = Path(self.config.model)
        self.model_name = self.model_path.stem
        self.device = config.training_args.device
        self.input_shape = config.input_shape
        # convert to list if not already
        if not isinstance(self.input_shape, list):
            self.input_shape = list(self.input_shape)
        # add batch dimension of one
        self.input_shape = [1] + self.input_shape

        # init components
        self.quantization_setup: Optional[QuantizationSetup] = None
        self.trainer: Optional[QuantizationAwareTrainer] = None

        # init model run meta data
        self.dataset_config: DataConfig = self._load_data_config()
        self.run_name = self._generate_run_name()
        #self.wandb_run = self._init_wandb()

    def run(self) -> None:
        logger.info("Starting QAT pipeline")

        logger.info("Setting up quantization")
        self._setup_quantization()

        logger.info("Running calibration")
        self._calibrate()

        logger.info("Configure training")
        self._initialize_trainer()

        logger.info("Preparing training")
        self._run_training_loop()

        logger.info("Training pipline completed successfully")

    def _setup_quantization(self) -> None:
        onnx_model_path = Path(self.config.onnx_model_path)

        self.quantization_setup = QuantizationSetup(
            onnx_path=onnx_model_path, device=self.device, quantization_settings=self.config.quantization_args
        )

        self.quantization_setup.load_model()
        self.quantization_setup.setup_quantizer()
        dispatching_table = self.quantization_setup.create_dispatching_table()

        self.quantization_setup.initialize_quantization(
            dispatching_table=dispatching_table, input_shape=self.input_shape
        )

    def _calibrate(self) -> None:
        if self.quantization_setup is None:
            raise RuntimeError("Quantization setup must be completed before calibration")

        calibration_dataset = CalibrationDataset(
            path=Path(self.config.calib_dataset_path),
            img_size=(self.input_shape[2], self.input_shape[3]),
        )

        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=1,  # only batch size of 1 is supported for calibration
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        calibration_pipeline = self.quantization_setup.create_calibration_pipeline()
        self.quantization_setup.run_calibration(calibration_dataloader, calibration_pipeline)

    def _initialize_trainer(self) -> None:
        self.trainer = QuantizationAwareTrainer(
            ppq_graph=self.quantization_setup.graph,  # type: ignore
            yolo_model=self.model_path,
            onnx_model_path=Path(self.config.onnx_model_path),
            dataset_yaml_path=Path(self.config.dataset_yaml_file_path),
            training_arguments=self.config.training_args,
            num_bits=self.config.quantization_args.num_bits,
        )

    def _load_data_config(self) -> DataConfig:
        """Loads and validates YAML config file for a YOLO dataset."""
        with open(self.config.dataset_yaml_file_path, "r") as f:
            raw_config = yaml.safe_load(f)
        try:
            return DataConfig(**raw_config)
        except ValidationError as e:
            logger.error("❌ Data config validation error:\n%s", e)
            raise SystemExit(1)
        except ValueError as e:
            logger.error("Invalid argument for data configuration: \n%s", e)
            raise SystemExit(1)

    def _run_training_loop(self) -> None:
        training_dataset = TrainDataset(
            path=Path(self.dataset_config.path),
            img_size=(self.input_shape[2], self.input_shape[3]),
            split=self.dataset_config.train,
        )
        training_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=1,  # only batch_size=1 is supported
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        output_dir = self._create_output_dir(self.run_name)

        logger.info(f"Start training for {self.config.training_args.epochs} epochs")
        for epoch in range(self.config.training_args.epochs):
            epoch_loss = self.trainer.train_epoch(training_dataloader)  # type: ignore

            epoch_model_name = f"qat_{self.model_name}_epoch_{epoch}"
            espdl_path = output_dir / "espdl" / f"{epoch_model_name}.espdl"
            native_path = output_dir / "native" / f"{epoch_model_name}.native"

            self.trainer.save_model(espdl_path, native_path)  # type: ignore

            # FIXME: model evaluation during training on validation dataset requires custom post-processing
            # metrics_path = output_dir / "metrics" / f"{epoch_model_name}_metrics.csv"
            # metrics = self.trainer.evaluate(self.config.save_metrics, metrics_path)

            # if self.trainer.update_metrics(metrics):
            #     best_espdl_model = output_dir / "espdl" / "best.espdl"
            #     best_native_model = output_dir / "native" / "best.native"
            #     self.trainer.save_model(best_espdl_model, best_native_model)
            #
            #     self._wandb_log_best_model(
            #         epoch=epoch,
            #         best_espdl_model_path=best_espdl_model,
            #         best_native_model_path=best_native_model
            #     )
            #
            # self._wandb_log_epoch(
            #     epoch=epoch,
            #     train_loss=epoch_loss,
            #     val_metrics=metrics,
            #     espdl_model_path=espdl_path,
            #     native_model_path=native_path
            # )
            logger.info(f"Epoch: {epoch + 1}: Loss: {epoch_loss:.4f}")

    def _generate_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"qat_{self.model_path.stem}_{timestamp}"

    @staticmethod
    def _create_output_dir(run_name: str) -> Path:
        output_dir = Path("qat-runs") / run_name
        for sub_dir in ["espdl", "native", "metrics"]:
            (output_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    # @staticmethod
    # def _wandb_login() -> None:
    #     #wandb.login(anonymous="allow", key=os.environ["WANDB_API_KEY"], timeout=60)

    # def _init_wandb(self) -> wandb.sdk.wandb_run.Run:
    #     return wandb.init(
    #         project=WANDB_PROJECT,
    #         name=self.run_name,
    #         tags=["QAT"],
    #         config=self.config.model_dump(),
    #         anonymous="allow",
    #         force=True,
    #     )

    # def _wandb_log_epoch(
    #     self,
    #     epoch: int,
    #     train_loss: float,
    #     val_metrics: DetMetrics,
    #     espdl_model_path: Path,
    #     native_model_path: Path,
    # ) -> None:
    #     """
    #     Logs epoch data to Weights & Biases
    #     :param epoch: current epoch
    #     :param train_loss: current training loss
    #     :param val_metrics: validation metrics
    #     :param espdl_model_path: path to .espdl model file of the current epoch
    #     :param native_model_path: path to .native model file of the current epoch
    #     """
    #     wandb.log(
    #         data={
    #             "train_loss": train_loss,
    #             "val_metrics": {
    #                 "curves": {
    #                     curve: curve_results
    #                     for curve, curve_results in zip(val_metrics.curves, val_metrics.curves_results)
    #                 },
    #                 **val_metrics.results_dict,
    #             }
    #             if val_metrics
    #             else {},
    #         },
    #         commit=True,
    #     )
    #     # log .espdl model
    #     artifact = wandb.Artifact(espdl_model_path.stem, type="model", metadata=self.config.model_dump())
    #     artifact.add_file(espdl_model_path.as_posix())
    #     # log .native model
    #     artifact.add_file(native_model_path.as_posix())
    #     self.wandb_run.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

    # def _wandb_log_best_model(self, epoch: int, best_espdl_model_path: Path, best_native_model_path: Path) -> None:
    #     """
    #     Logs/overwrites the best-performing model of the current model run to Weights & Biases
    #     :param epoch: current epoch
    #     :param best_espdl_model_path: path to best .espdl model file
    #     :param best_native_model_path: path to best .native model file
    #     """
    #     # log .espdl model
    #     artifact = wandb.Artifact(
    #         best_espdl_model_path.stem,
    #         type="model",
    #         metadata=self.config.model_dump(),
    #     )
    #     artifact.add_file(best_espdl_model_path.as_posix())
    #     artifact.add_file(best_native_model_path.as_posix())
    #     self.wandb_run.log_artifact(best_espdl_model_path, aliases=["best", f"epoch_{epoch}"])
    #
    # def finish_wandb(self) -> None:
    #     wandb.finish()
    #     logger.info(f"Training completed, view results under: {self.wandb_run.url}")
