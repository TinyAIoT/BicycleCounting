import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from ppq.api import load_native_graph
from ppq.executor import TorchExecutor
from ultralytics import YOLO
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    de_parallel,
    select_device,
    smart_inference_mode,
)

from utils.quantization import quantize_yolo


class QuantizedModelValidator(DetectionValidator):
    def __init__(self, args=None, _callbacks=None) -> None:
        super().__init__()

        self.args = args
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # Initialize required attributes
        # self.metrics = {}
        self.speed: dict[Any, Any] = {}
        self.jdict: list[Any] = []
        self.loss = 0
        self.training = False

    @staticmethod
    def ppq_graph_init(quant_func: Callable, device, native_path: Optional[Path] = None, **kwargs):
        """
        Init ppq graph inference.
            # case 1: PTQ graph validation: ppq_graph = quant_func()
            # case 2: QAT graph validation:
                        utilize .native to load the graph
                        while training, the .native model is saved along with .espdl model
        """
        if native_path:
            ppq_graph = load_native_graph(native_path.as_posix())
        else:
            ppq_graph = quant_func(**kwargs, device=device)

        executor = TorchExecutor(graph=ppq_graph, device=device)
        return executor

    @staticmethod
    def ppq_graph_inference(executor, task, inputs, device):
        """ppq graph inference"""
        graph_outputs = executor(inputs)
        if task == "detect":
            if len(graph_outputs) > 1:
                x = [torch.cat((graph_outputs[i], graph_outputs[i + 1]), 1) for i in range(0, len(graph_outputs), 2)]
            else:
                x = graph_outputs
            detect_model = Detect(nc=80, ch=[32, 64, 128])
            detect_model.stride = [8.0, 16.0, 32.0]
            detect_model.to(device)

            y = detect_model._inference(x)
            return y
        else:
            raise NotImplementedError(f"{task} is not supported.")

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, *args, **kwargs):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        # Set default arguments if none provided
        native_model_path = self.args.get("native_model_path")
        onnx_model_path = self.args.get("onnx_model_path")
        num_bits = self.args.get("num_bits")

        override_args = kwargs.get("args", {})
        override_args.update(dict(batch=1))
        from ultralytics.cfg import get_cfg

        self.args = get_cfg(overrides=override_args)

        # self.args.data = kwargs.get('args', {}).get('data', None)
        # self.training = trainer is not None
        self.training = None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.args.amp
            model = YOLO(kwargs.get("args", None).get("model", None))
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer._epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)

            model = AutoBackend(
                weights=model or self.args.model or kwargs.get("args", {}).get("model", None),
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data or kwargs.get("args", {}).get("data", None),
                fp16=self.args.half,
            )

            # self.model = model
            self.device = model.device  # update device

            self.args.half = model.fp16  # update half

            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading

            if not pt:
                self.args.rect = False  # set to false

            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), 1)

            model.eval()
            model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        executor = self.ppq_graph_init(
            quantize_yolo,
            device="cpu",
            native_path=native_model_path,
            onnx_model_path=onnx_model_path,
            num_of_bits=num_bits,
        )

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            batch["img"] = F.interpolate(batch["img"], size=(640, 640), mode="bilinear", align_corners=False)
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                preds = self.ppq_graph_inference(executor, "detect", batch["img"], "cpu")
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

    # def get_dataloader(self, dataset_path, batch_size):
    #     """
    #     Construct and return dataloader.
    #
    #     Args:
    #         dataset_path (str): Path to the dataset.
    #         batch_size (int): Size of each batch.
    #
    #     Returns:
    #         (torch.utils.data.DataLoader): Dataloader for validation.
    #     """
    #     dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    #     return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    #
    # def build_dataset(self, img_path, mode="val", batch=None):
    #     """
    #     Build YOLO Dataset.
    #
    #     Args:
    #         img_path (str): Path to the folder containing images.
    #         mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
    #         batch (int, optional): Size of batches, this is for `rect`.
    #
    #     Returns:
    #         (Dataset): YOLO dataset.
    #     """
    #     return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
    #
    # def preprocess(self, batch):
    #     """
    #     Preprocess batch of images for YOLO validation.
    #
    #     Args:
    #         batch (dict): Batch containing images and annotations.
    #
    #     Returns:
    #         (dict): Preprocessed batch.
    #     """
    #     batch["img"] = batch["img"].to(self.device, non_blocking=True)
    #     batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
    #     for k in ["batch_idx", "cls", "bboxes"]:
    #         batch[k] = batch[k].to(self.device)
    #
    #     return batch
    #
    # def postprocess(self, preds):
    #     """
    #     Apply Non-maximum suppression to prediction outputs.
    #
    #     Args:
    #         preds (torch.Tensor): Raw predictions from the model.
    #
    #     Returns:
    #         (List[torch.Tensor]): Processed predictions after NMS.
    #     """
    #     return non_max_suppression(
    #         preds,
    #         self.args.conf,
    #         self.args.iou,
    #         nc=0 if self.args.task == "detect" else self.nc,
    #         multi_label=True,
    #         agnostic=self.args.single_cls or self.args.agnostic_nms,
    #         max_det=self.args.max_det,
    #         end2end=self.end2end,
    #         rotated=self.args.task == "obb",
    #     )
    #
    # def update_metrics(self, preds, batch):
    #     """
    #     Update metrics with new predictions and ground truth.
    #
    #     Args:
    #         preds (List[torch.Tensor]): List of predictions from the model.
    #         batch (dict): Batch data containing ground truth.
    #     """
    #     for si, pred in enumerate(preds):
    #         self.seen += 1
    #         npr = len(pred)
    #         stat = dict(
    #             conf=torch.zeros(0, device=self.device),
    #             pred_cls=torch.zeros(0, device=self.device),
    #             tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
    #         )
    #         pbatch = self._prepare_batch(si, batch)
    #         cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    #         nl = len(cls)
    #         stat["target_cls"] = cls
    #         stat["target_img"] = cls.unique()
    #         if npr == 0:
    #             if nl:
    #                 for k in self.stats.keys():
    #                     self.stats[k].append(stat[k])
    #                 if self.args.plots:
    #                     self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
    #             continue
    #
    #         # Predictions
    #         if self.args.single_cls:
    #             pred[:, 5] = 0
    #         predn = self._prepare_pred(pred, pbatch)
    #         stat["conf"] = predn[:, 4]
    #         stat["pred_cls"] = predn[:, 5]
    #
    #         # Evaluate
    #         if nl:
    #             stat["tp"] = self._process_batch(predn, bbox, cls)
    #         if self.args.plots:
    #             self.confusion_matrix.process_batch(predn, bbox, cls)
    #         for k in self.stats.keys():
    #             self.stats[k].append(stat[k])
    #
    # def _process_batch(self, detections, gt_bboxes, gt_cls):
    #     """
    #     Return correct prediction matrix.
    #
    #     Args:
    #         detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
    #             (x1, y1, x2, y2, conf, class).
    #         gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
    #             bounding box is of the format: (x1, y1, x2, y2).
    #         gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.
    #
    #     Returns:
    #         (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
    #     """
    #     iou = box_iou(gt_bboxes, detections[:, :4])
    #     return self.match_predictions(detections[:, 5], gt_cls, iou)
    #
    # def _prepare_pred(self, pred, pbatch):
    #     """
    #     Prepare predictions for evaluation against ground truth.
    #
    #     Args:
    #         pred (torch.Tensor): Model predictions.
    #         pbatch (dict): Prepared batch information.
    #
    #     Returns:
    #         (torch.Tensor): Prepared predictions in native space.
    #     """
    #     predn = pred.clone()
    #     scale_boxes(
    #         pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
    #     )  # native-space pred
    #     return predn


class QuantDetectionValidator(QuantizedModelValidator):
    def __init__(self, args=None, _callbacks=None):
        super().__init__(args=args, _callbacks=_callbacks)

    def __call__(self, *args, **kwargs):
        return super().__call__(self, *args, **kwargs)
