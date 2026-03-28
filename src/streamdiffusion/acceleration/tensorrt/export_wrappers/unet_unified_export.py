import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import UNet2DConditionModel

from .unet_controlnet_export import create_controlnet_wrapper
from .unet_ipadapter_export import create_ipadapter_wrapper
from ..models.utils import convert_list_to_structure

logger = logging.getLogger(__name__)


class UnifiedExportWrapper(torch.nn.Module):
    """
    Unified wrapper that composes conditioning modules for UNet export.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        use_controlnet: bool = False,
        use_ipadapter: bool = False,
        control_input_names: Optional[List[str]] = None,
        num_tokens: int = 4,
        ipadapter_layer_policy: str = "all",
        kvo_cache_structure: List[int] = [],
        fused_controlnets: Optional[List[torch.nn.Module]] = None,
        fused_controlnet_conditioning_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.use_controlnet = use_controlnet
        self.use_ipadapter = use_ipadapter
        self.controlnet_wrapper = None
        self.ipadapter_wrapper = None
        self.unet = unet
        self.kvo_cache_structure = kvo_cache_structure

        fused_controlnets = fused_controlnets or []
        self.fused_controlnets = torch.nn.ModuleList(fused_controlnets)
        self.fused_controlnet_conditioning_channels = list(
            fused_controlnet_conditioning_channels or []
        )
        self.use_fused_controlnet = len(self.fused_controlnets) > 0
        self._expects_sdxl_added_cond = False

        if use_ipadapter:
            ipadapter_kwargs = {
                k: v for k, v in kwargs.items() if k in ["install_processors"]
            }
            if "install_processors" not in ipadapter_kwargs:
                ipadapter_kwargs["install_processors"] = True

            self.ipadapter_wrapper = create_ipadapter_wrapper(
                unet,
                num_tokens=num_tokens,
                layer_policy=ipadapter_layer_policy,
                **ipadapter_kwargs,
            )
            self.unet = self.ipadapter_wrapper.unet

        self._align_fused_controlnets_to_unet()
        self._expects_sdxl_added_cond = self._detect_sdxl_added_cond_support(self.unet)

        if use_controlnet and control_input_names and not self.use_fused_controlnet:
            controlnet_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["num_controlnets", "conditioning_scales"]
            }
            self.controlnet_wrapper = create_controlnet_wrapper(
                self.unet,
                control_input_names,
                kvo_cache_structure,
                **controlnet_kwargs,
            )

    def _align_fused_controlnets_to_unet(self) -> None:
        if not self.use_fused_controlnet:
            return

        try:
            ref_param = next(self.unet.parameters())
            target_device = ref_param.device
            target_dtype = ref_param.dtype
        except Exception:
            target_device = None
            target_dtype = None

        for controlnet in self.fused_controlnets:
            try:
                move_kwargs = {}
                if target_device is not None:
                    move_kwargs["device"] = target_device
                if target_dtype is not None:
                    move_kwargs["dtype"] = target_dtype
                if move_kwargs:
                    controlnet.to(**move_kwargs)
            except Exception:
                logger.debug(
                    "UnifiedExportWrapper: could not pre-align fused ControlNet device/dtype",
                    exc_info=True,
                )

    def _detect_sdxl_added_cond_support(self, module: torch.nn.Module) -> bool:
        candidate = module
        for attr_name in ["unet", "unet_model"]:
            if hasattr(candidate, attr_name):
                next_candidate = getattr(candidate, attr_name)
                if hasattr(next_candidate, "config"):
                    candidate = next_candidate
                    break

        config = getattr(candidate, "config", None)
        return getattr(config, "addition_embed_type", None) == "text_time"

    def _basic_unet_forward(
        self, sample, timestep, encoder_hidden_states, *kvo_cache, **kwargs
    ):
        formatted_kvo_cache = []
        if len(kvo_cache) > 0:
            formatted_kvo_cache = convert_list_to_structure(
                kvo_cache, self.kvo_cache_structure
            )

        unet_kwargs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
            **kwargs,
        }
        if len(kvo_cache) > 0:
            unet_kwargs["kvo_cache"] = formatted_kvo_cache
        res = self.unet(**unet_kwargs)
        if len(kvo_cache) > 0:
            return res
        return res[0]

    def _extract_sdxl_added_cond(
        self,
        sample: torch.Tensor,
        args: Tuple[torch.Tensor, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Tuple[torch.Tensor, ...]]:
        added_cond_kwargs = kwargs.pop("added_cond_kwargs", None)
        if added_cond_kwargs is not None:
            return added_cond_kwargs, args

        if not self._expects_sdxl_added_cond:
            return None, args

        if len(args) >= 2:
            return {
                "text_embeds": args[0],
                "time_ids": args[1],
            }, args[2:]

        batch_size = sample.shape[0]
        return {
            "text_embeds": torch.zeros(
                batch_size, 1280, device=sample.device, dtype=sample.dtype
            ),
            "time_ids": torch.zeros(
                batch_size, 6, device=sample.device, dtype=sample.dtype
            ),
        }, args

    def _extract_fused_control_inputs(
        self,
        sample: torch.Tensor,
        args: Tuple[torch.Tensor, ...],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, ...]]:
        control_images: List[torch.Tensor] = []
        control_scales: List[torch.Tensor] = []
        remaining_args = list(args)
        batch_size = sample.shape[0]
        image_height = sample.shape[2] * 8
        image_width = sample.shape[3] * 8

        for idx in range(len(self.fused_controlnets)):
            if len(remaining_args) >= 2:
                control_images.append(remaining_args.pop(0))
                control_scales.append(remaining_args.pop(0))
                continue

            channels = 3
            if idx < len(self.fused_controlnet_conditioning_channels):
                channels = self.fused_controlnet_conditioning_channels[idx]
            control_images.append(
                torch.zeros(
                    batch_size,
                    channels,
                    image_height,
                    image_width,
                    device=sample.device,
                    dtype=sample.dtype,
                )
            )
            control_scales.append(
                torch.tensor(0.0, device=sample.device, dtype=torch.float32)
            )

        return control_images, control_scales, tuple(remaining_args)

    def _run_fused_controlnets(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_images: List[torch.Tensor],
        control_scales: List[torch.Tensor],
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        merged_down = None
        merged_mid = None

        for idx, controlnet in enumerate(self.fused_controlnets):
            controlnet_kwargs = {
                "sample": sample,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "controlnet_cond": control_images[idx],
                "conditioning_scale": control_scales[idx],
                "return_dict": False,
            }
            if added_cond_kwargs is not None:
                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            down_samples, mid_sample = controlnet(**controlnet_kwargs)

            if merged_down is None:
                merged_down = list(down_samples)
                merged_mid = mid_sample
                continue

            for down_idx, down_sample in enumerate(down_samples):
                merged_down[down_idx] = merged_down[down_idx] + down_sample
            merged_mid = merged_mid + mid_sample

        return merged_down, merged_mid

    def _forward_with_fused_controlnet(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        control_images, control_scales, remaining_args = self._extract_fused_control_inputs(
            sample, args
        )
        added_cond_kwargs, remaining_args = self._extract_sdxl_added_cond(
            sample, remaining_args, kwargs
        )
        formatted_kvo_cache = []
        if len(remaining_args) > 0:
            formatted_kvo_cache = convert_list_to_structure(
                remaining_args, self.kvo_cache_structure
            )

        merged_down, merged_mid = self._run_fused_controlnets(
            sample,
            timestep,
            encoder_hidden_states,
            control_images,
            control_scales,
            added_cond_kwargs,
        )

        unet_kwargs: Dict[str, Any] = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }
        if added_cond_kwargs is not None:
            unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
        if len(formatted_kvo_cache) > 0:
            unet_kwargs["kvo_cache"] = formatted_kvo_cache
        if merged_down is not None:
            unet_kwargs["down_block_additional_residuals"] = merged_down
        if merged_mid is not None:
            unet_kwargs["mid_block_additional_residual"] = merged_mid
        if kwargs:
            unet_kwargs.update(kwargs)

        res = self.unet(**unet_kwargs)
        if len(formatted_kvo_cache) > 0:
            return res
        return res[0]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_ipadapter and self.ipadapter_wrapper is not None:
            if len(args) == 0:
                logger.error(
                    "UnifiedExportWrapper: ipadapter_scale missing; required when use_ipadapter=True"
                )
                raise RuntimeError(
                    "UnifiedExportWrapper: ipadapter_scale tensor is required when use_ipadapter=True"
                )
            ipadapter_scale = args[0]
            if not isinstance(ipadapter_scale, torch.Tensor):
                logger.error(
                    "UnifiedExportWrapper: ipadapter_scale wrong type: %s",
                    type(ipadapter_scale),
                )
                raise TypeError("ipadapter_scale must be a torch.Tensor")
            self.ipadapter_wrapper.set_ipadapter_scale(ipadapter_scale)
            args = args[1:]

        if self.use_fused_controlnet:
            return self._forward_with_fused_controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        if self.controlnet_wrapper:
            return self.controlnet_wrapper(
                sample, timestep, encoder_hidden_states, *args, **kwargs
            )

        return self._basic_unet_forward(
            sample, timestep, encoder_hidden_states, *args, **kwargs
        )
