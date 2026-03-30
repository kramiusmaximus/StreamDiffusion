import gc
import os
from typing import *

import torch

from .models.models import BaseModel
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        min_image_height=None,
        max_image_height=None,
        min_image_width=None,
        max_image_width=None,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = True,
        build_all_tactics: bool = False,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
        trt_precision: str = "fp16",
    ):
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            self.network = self.network.to("cpu")
            del self.network
            gc.collect()
            torch.cuda.empty_cache()
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached model: {onnx_opt_path}")
        else:
            print(f"Generating optimizing model: {onnx_opt_path}")
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
        min_image_height = min_image_resolution if min_image_height is None else min_image_height
        max_image_height = max_image_resolution if max_image_height is None else max_image_height
        min_image_width = min_image_resolution if min_image_width is None else min_image_width
        max_image_width = max_image_resolution if max_image_width is None else max_image_width

        self.model.min_image_height = min_image_height
        self.model.max_image_height = max_image_height
        self.model.min_image_width = min_image_width
        self.model.max_image_width = max_image_width
        self.model.min_latent_height = min_image_height // 8
        self.model.max_latent_height = max_image_height // 8
        self.model.min_latent_width = min_image_width // 8
        self.model.max_latent_width = max_image_width // 8

        # Preserve legacy scalar attributes for callers that still expect them.
        self.model.min_image_shape = min(min_image_height, min_image_width)
        self.model.max_image_shape = max(max_image_height, max_image_width)
        self.model.min_latent_shape = min(self.model.min_latent_height, self.model.min_latent_width)
        self.model.max_latent_shape = max(self.model.max_latent_height, self.model.max_latent_width)
        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                trt_precision=trt_precision,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
            )
        for file in os.listdir(os.path.dirname(engine_path)):
            if file.endswith('.engine'):
                continue
            os.remove(os.path.join(os.path.dirname(engine_path), file))
        
        gc.collect()
        torch.cuda.empty_cache()
