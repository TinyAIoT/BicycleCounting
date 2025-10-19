import logging
from pathlib import Path
from typing import Literal, Optional, Sequence

import onnx
import ppq.lib as PFL
import torch
from onnxsim import simplify
from ppq import QuantizationSettingFactory
from ppq.api import quantize_onnx_model
from ppq.api.interface import load_onnx_graph
from ppq.core import TargetPlatform
from ppq.core.quant import QuantizationVisibility
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph
from ppq.quantization.optim.calibration import RuntimeCalibrationPass
from ppq.quantization.optim.parameters import (
    ParameterQuantizePass,
    PassiveParameterQuantizePass,
)
from ppq.quantization.optim.refine import (
    QuantAlignmentPass,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
)
from torch.utils.data import DataLoader

from core.constants import TARGET_PLATFORM
from core.schemas import QuantizationArgs
from utils.datasets import CalibrationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuantizationSetup:
    """Auxiliary class handling quantization setup and calibration"""

    def __init__(
        self, onnx_path: Path, device: Literal["cpu", "cuda"], quantization_settings: QuantizationArgs
    ) -> None:
        """Init quantization setup"""
        self.onnx_path = onnx_path
        self.device = device
        self.num_bits = quantization_settings.num_bits
        self.calib_steps = quantization_settings.calib_steps
        self.dispatching_override = quantization_settings.dispatching_override
        self.graph: Optional[BaseGraph] = None
        self.executor: Optional[TorchExecutor] = None
        self.quantizer: Optional[PFL.Quantizer] = None

    def load_model(self) -> None:
        """Load ONNX model from path"""
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path.as_posix()}")
        self.graph = load_onnx_graph(onnx_import_file=self.onnx_path.as_posix())

    def setup_quantizer(self) -> None:
        """Setup quantizer for ESP32-S3"""
        if not self.graph:
            raise ValueError("Graph must be set before quantizer setup")

        platform = target_platform(TARGET_PLATFORM, self.num_bits)
        self.quantizer = PFL.Quantizer(platform=platform, graph=self.graph)

    def create_dispatching_table(self) -> dict[str, TargetPlatform]:
        """
        Create and configure dispatching table
        :return: altered dispatching table
        """
        if not (self.quantizer and self.graph):
            raise ValueError("Quantizer and graph must be initialized")

        dispatcher = PFL.Dispatcher(graph=self.graph, method="conservative")
        dispatching_table = dispatcher.dispatch(self.quantizer.quant_operation_types)

        # override dispatching result
        if self.dispatching_override:
            for opname, platform in self.dispatching_override.items():
                if opname not in self.graph.operations:
                    continue
                assert isinstance(platform, int) or isinstance(platform, TargetPlatform), (
                    f"Your dispatching_override table contains a invalid setting of operation {opname}, "
                    "All platform setting given in dispatching_override table is expected given as int or "
                    "TargetPlatform, however {type(platform)} was given."
                )
                dispatching_table[opname] = TargetPlatform(platform)

        for opname, platform in dispatching_table.items():
            if platform == TargetPlatform.UNSPECIFIED:
                dispatching_table[opname] = TargetPlatform(self.quantizer.target_platform)

        return dispatching_table

    def initialize_quantization(
        self,
        dispatching_table: dict[str, TargetPlatform],
        input_shape: Sequence[int],
    ) -> TorchExecutor:
        """
        Initializes the quantization operations and prepares TorchExecutor
        :param dispatching_table: created dispatching table
        :param input_shape: Model input shape
        :return: TorchExecutor instance
        """
        if not (self.graph and self.quantizer):
            raise ValueError("Graph and quantizer must be initialized")

        # Initialize quantization for each operation
        for op in self.graph.operations.values():
            self.quantizer.quantize_operation(op_name=op.name, platform=dispatching_table[op.name])

        # Create executor and trace operations
        self.executor = TorchExecutor(graph=self.graph, device=self.device)
        dummy_input = torch.zeros(input_shape).to(self.device)
        self.executor.tracing_operation_meta(inputs=dummy_input)

        return self.executor

    def create_calibration_pipeline(self) -> PFL.Pipeline:
        """Create calibration pipeline."""
        if self.quantizer is None:
            raise ValueError("Quantizer must be initialized")

        return PFL.Pipeline(
            [
                QuantizeSimplifyPass(),
                QuantizeFusionPass(activation_type=self.quantizer.activation_fusion_types),
                ParameterQuantizePass(),
                RuntimeCalibrationPass(method="kl", calib_steps=self.calib_steps),
                PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE),
                QuantAlignmentPass(elementwise_alignment="Align to Output"),
            ]
        )

    def run_calibration(self, calibration_dataloader: DataLoader, pipeline: PFL.Pipeline) -> None:
        """Run calibration process."""
        if self.graph is None or self.executor is None:
            raise ValueError("Graph and executor must be initialized")

        def collate_fn(x: torch.Tensor) -> torch.Tensor:
            return x.type(torch.float).to(self.device)

        pipeline.optimize(
            calib_steps=self.calib_steps,
            collate_fn=collate_fn,
            graph=self.graph,
            dataloader=calibration_dataloader,
            executor=self.executor,
        )

        logger.info(f"Calibration completed with {len(calibration_dataloader.dataset)} images")  # type: ignore


def quantize_yolo(
    onnx_model_path: Path,
    espdl_model_path: Path,
    calib_dataset_path: Path,
    num_of_bits: Literal[8, 16] = 8,
    calib_steps: int = 32,
    sim: bool = True,
    device: Literal["cpu", "cuda"] = "cpu",
) -> BaseGraph:
    """
    On-the-fly quantization of YOLO11n for Quantization-aware Training
    :param onnx_model_path: Path to ONNX model file
    :param espdl_model_path: Path to export ESP-DL model file
    :param calib_dataset_path: Path to calibration dataset
    :param input_shape: Model input shape of form [C, H, W]
    :param batch_size: Batch size used for quantization
    :param num_of_bits: Number of bits used for quantization. Only int8 and int16 are supported for .espdl
    :param calib_steps: Number of calibration steps
    :param sim: Whether to simplify the ONNX graph
    :param device: Device used for quantization. Only cpu and cuda are supported.
    :return: quantized graph from ESP-PPQ framework.
    """
    if num_of_bits not in (8, 16):
        raise ValueError(f"int8 and int16 are supported but received {num_of_bits} bit for quantization.")
    # validate onnx_model_path
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"No such path: {onnx_model_path.as_posix()}")
    if not onnx_model_path.suffix == ".onnx":
        raise IOError(f"Invalid file format for onnx_model_path: {onnx_model_path.suffix}. Expected .onnx")
    # validate espdl_model_path
    if not espdl_model_path.suffix == ".espdl":
        raise IOError(f"Invalid file format for espdl_model_path: {espdl_model_path.suffix}. Expected .espdl")
    # validate calibration dataset path
    if not (calib_dataset_path.is_dir() and calib_dataset_path.exists()):
        raise NotADirectoryError(f"{calib_dataset_path.as_posix()} is not a directory or does not exist.")

    model = onnx.load(onnx_model_path.as_posix())
    if sim:
        model, check = simplify(model)
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_model_path.as_posix())

    calibration_dataset = CalibrationDataset(calib_dataset_path)
    dataloader = DataLoader(dataset=calibration_dataset, batch_size=1, shuffle=False)

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(device)

    # default setting
    quant_setting = QuantizationSettingFactory.espdl_setting()

    """
    # Mixed-Precision + Horizontal Layer Split Pass Quantization

    quant_setting.dispatching_table.append(
        operation='/model.2/cv2/conv/Conv',
        platform=target_platform(TARGET_PLATFORM, 16)
    )
    quant_setting.dispatching_table.append(
        operation='/model.3/conv/Conv',
        platform=target_platform(TARGET_PLATFORM, 16)
    )

    quant_setting.dispatching_table.append(
        operation='/model.4/cv2/conv/Conv',
        platform=target_platform(TARGET_PLATFORM, 16)
    )

    quant_setting.weight_split = True
    quant_setting.weight_split_setting.method = 'balance'
    quant_setting.weight_split_setting.value_threshold = 1.5 #1.5
    quant_setting.weight_split_setting.interested_layers = ['/model.0/conv/Conv',
                                                            '/model.1/conv/Conv' ]
    """
    x = calibration_dataset[0].unsqueeze(0)

    quant_ppq_graph = quantize_onnx_model(
        onnx_import_file=onnx_model_path.as_posix(),
        espdl_export_file=espdl_model_path.as_posix(),
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=x.shape,
        target=TARGET_PLATFORM,
        num_of_bits=num_of_bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph
