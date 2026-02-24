"""StreamDiffusionTD Local Operator Extension"""
import os
import subprocess
import socket
import json
import time
import traceback
import asyncio
from functools import reduce
import ast
import datetime
import numpy as np
import shutil
import re
from urllib.parse import urlparse
from TDStoreTools import StorageManager
import TDFunctions as TDF
import platform
import copy

#region CONFIG - Model registries and parameter mappings

CN_MODEL_REGISTRY = {
    'sd15': [
        {'id': 'lllyasviel/control_v11f1p_sd15_depth', 'name': 'Depth', 'type': 'depth', 'compatibility': ['local', 'daydream']},
        {'id': 'lllyasviel/control_v11f1e_sd15_tile', 'name': 'Tile', 'type': 'tile', 'compatibility': ['local', 'daydream']},
        {'id': 'lllyasviel/control_v11p_sd15_canny', 'name': 'Canny', 'type': 'canny', 'compatibility': ['local', 'daydream']},
        {'id': 'lllyasviel/control_v11p_sd15_openpose', 'name': 'OpenPose', 'type': 'openpose', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_mlsd', 'name': 'MLSD', 'type': 'mlsd', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_normalbae', 'name': 'Normal', 'type': 'normalbae', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_scribble', 'name': 'Scribble', 'type': 'scribble', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_seg', 'name': 'Segmentation', 'type': 'seg', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_lineart', 'name': 'Lineart', 'type': 'lineart', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15s2_lineart_anime', 'name': 'Lineart Anime', 'type': 'lineart_anime', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11e_sd15_shuffle', 'name': 'Shuffle', 'type': 'shuffle', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_inpaint', 'name': 'Inpaint', 'type': 'inpaint', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11e_sd15_ip2p', 'name': 'IP2P', 'type': 'ip2p', 'compatibility': ['local']},
        {'id': 'lllyasviel/control_v11p_sd15_softedge', 'name': 'Soft Edge', 'type': 'softedge', 'compatibility': ['local']}
    ],
    'sd21': [
        {'id': 'thibaud/controlnet-sd21-openpose-diffusers', 'name': 'OpenPose', 'type': 'openpose', 'compatibility': ['local', 'daydream']},
        {'id': 'thibaud/controlnet-sd21-hed-diffusers', 'name': 'HED', 'type': 'hed', 'compatibility': ['local', 'daydream']},
        {'id': 'thibaud/controlnet-sd21-canny-diffusers', 'name': 'Canny', 'type': 'canny', 'compatibility': ['local', 'daydream']},
        {'id': 'thibaud/controlnet-sd21-depth-diffusers', 'name': 'Depth', 'type': 'depth', 'compatibility': ['local', 'daydream']},
        {'id': 'thibaud/controlnet-sd21-color-diffusers', 'name': 'Color', 'type': 'color', 'compatibility': ['local', 'daydream']},
        {'id': 'thibaud/controlnet-sd21-scribble-diffusers', 'name': 'Scribble', 'type': 'scribble', 'compatibility': ['local']},
        {'id': 'thibaud/controlnet-sd21-lineart-diffusers', 'name': 'Lineart', 'type': 'lineart', 'compatibility': ['local']},
        {'id': 'thibaud/controlnet-sd21-normalbae-diffusers', 'name': 'Normal', 'type': 'normalbae', 'compatibility': ['local']},
        {'id': 'thibaud/controlnet-sd21-zoedepth-diffusers', 'name': 'ZoeDepth', 'type': 'zoedepth', 'compatibility': ['local']},
        {'id': 'thibaud/controlnet-sd21-ade20k-diffusers', 'name': 'Segmentation', 'type': 'ade20k', 'compatibility': ['local']}
    ],
    'sdxl': [
        {'id': 'xinsir/controlnet-depth-sdxl-1.0', 'name': 'Depth', 'type': 'depth', 'compatibility': ['local', 'daydream']},
        {'id': 'xinsir/controlnet-canny-sdxl-1.0', 'name': 'Canny', 'type': 'canny', 'compatibility': ['local', 'daydream']},
        {'id': 'xinsir/controlnet-tile-sdxl-1.0', 'name': 'Tile', 'type': 'tile', 'compatibility': ['local', 'daydream']},
        {'id': 'xinsir/controlnet-openpose-sdxl-1.0', 'name': 'OpenPose', 'type': 'openpose', 'compatibility': ['local']},
        {'id': 'xinsir/controlnet-scribble-sdxl-1.0', 'name': 'Scribble', 'type': 'scribble', 'compatibility': ['local']},
        {'id': 'diffusers/controlnet-depth-sdxl-1.0', 'name': 'Depth (Official)', 'type': 'depth', 'compatibility': ['local']}
    ]
}

DAYDREAM_SUPPORTED_MODELS = [
    'stabilityai/sd-turbo',
    'stabilityai/sdxl-turbo',
    'prompthero/openjourney-v4'
]


param_json_map = {
    "Modelid": "model_id_or_path",
    "Tindexlist": "t_index_list",
    "Promptdict0concept": "prompt",
    "Negprompt": "negative_prompt",
    "Framesize": "frame_buffer_size",
    "Width": "width",
    "Height": "height",
    "Acceleration": "acceleration",
    "Denoisebatch": "use_denoising_batch",
    "Seeddict0seedval": "seed",
    "Cfgtype": "cfg_type",
    "Guidancescale": "guidance_scale",
    "Delta": "delta",
    "Addnoise": "do_add_noise",
    "Imagefilter": "enable_similar_image_filter",
    "Filterthresh": "similar_image_filter_threshold",
    "Maxskipframe": "similar_image_filter_max_skip_frame",
    "Oscinport": "osc_in_port",
    "Oscoutport": "osc_out_port",
    "Sdmode":"sdmode",
    "Customlcm" : "lcm_lora_id",
    "Customvae" : "vae_id",
    "Streamoutname" : "input_mem_name",
    # "Limitfps": "max_fps",
    "Hfcache": "hf_cache",
    "Ipadapterenable": "use_ipadapter",
    "Ipadapterscale": "ipadapter_scale",
    "Addnoise": "do_add_noise",
    "Warmup": "warmup",
    "Safetychecker": "use_safety_checker",
    "Skipdiffusion": "skip_diffusion",
    "Compileengines": "compile_engines_only",
    "Buildifmissing": "build_engines_if_missing",
    # Scheduler/Sampler (init-time, requires restart)
    "Scheduler": "scheduler",
    "Sampler": "sampler",
    # StreamV2V Cached Attention (KVO Cache)
    "Cattenable": "use_cached_attn",      # Master toggle (init-time)
    "Cattmaxframes": "cache_maxframes",   # Live-updateable when enabled
    "Cattinterval": "cache_interval",     # Live-updateable when enabled
}
model_preset_params = [
    "Modelid",
    "Customlcm",
    "Loradictblock",
    "Uselora",
    "Usecustomlcm",
    "Acceleration",
    "Tindexblock",
    "Skiplcm",
    "Width",
    "Height",
]

#endregion CONFIG

class StreamDiffusionExt:
    """Local GPU backend for StreamDiffusion real-time image generation."""

    # Registry identifiers
    TOX_NAME = 'streamdiffusiontd'
    BRANCH = 'Local'

    def __init__(self, ownerComp):
        """
        Initializes the StarterExt class. Sets the owner component and calls the functions to create basic parameters and setup tables.
        """
        self.ownerComp = ownerComp
        self.message_box_open = False
        self.last_opened_time = time.time()
        self.logger = op('Logger').ext.Logger

        self.setup_project_info_table()
        # self.setup_par_details_table()
        self.Showbuiltin()
        self.Sethfcache()

        self.ownerComp.par.Streamactive = False
        self.ownerComp.par.Serveractive = False

        self.last_stop_time = time.time()
        self.Basefolder(force = True)
        self.ismac = platform.system() == 'Darwin'
        self.last_processframe_time = 0
        self.processframe_debounce_ms = 40
        self.last_feedback_safe_state = None
        self._in_resolution_change = False
        self.feedback_safe_paused_backend = False  # Track if we paused the backend
        self.last_feedback_safe_disable_time = 0  # Track when feedback safe was last disabled
        self.last_fx_sequence_size = self.ownerComp.par.Fx.sequence.numBlocks
        self.update_cn_id_menus()
        self.logger.log("StreamDiffusionTD Loaded", level='INFO')

        # Update checker state
        self._pending_update_version = None
        self._pending_update_path = None
        self._init_update_checker()

        # Check for restore data from DotToxLoader (if loaded from stub)
        self._check_and_restore_from_dotloader()

        # Register with TOX registry
        self._configure_registry()

    def get_version(self):
        return self.ownerComp.par.Txversion.eval()

    def get_subprocess_creationflags(self):
        td_version = self.get_td_version()
        if td_version and td_version >= 2025:
            return subprocess.CREATE_NEW_CONSOLE
        return 0

    def get_td_version(self):
        """
        Returns the TouchDesigner major version (e.g., 2023 or 2025).
        """
        build_year = int(app.build.split('.')[0])
        return build_year

    #region PARAMETER CALLBACKS
    def Backend(self):
        """Called when Backend parameter changes (TD automatic callback)"""
        target_backend = self.ownerComp.par.Backend.eval()

        # If switching away from Local, swap to that backend's TOX
        if target_backend != self.BRANCH:
            try:
                dotloader = self.ownerComp.op('dotloader')
                if dotloader and hasattr(dotloader, 'ext') and hasattr(dotloader.ext, 'DotToxLoaderExt'):
                    loader_ext = dotloader.ext.DotToxLoaderExt
                    tox_path = loader_ext.get_swap_target_path(target_backend, tox_name=self.TOX_NAME)
                    if tox_path:
                        print(f'[StreamDiffusionExt] Switching to {target_backend}...')
                        loader_ext.swap_operator(tox_path)
                        return
                    else:
                        print(f'[StreamDiffusionExt] {target_backend} not available. Install from registry.')
                        self.ownerComp.par.Backend = self.BRANCH
                        return
            except Exception as e:
                print(f'[StreamDiffusionExt] Error switching backend: {e}')
                self.ownerComp.par.Backend = self.BRANCH
                return

        backend = target_backend
        self.update_model_id_menu(backend)
        if self.ownerComp.par.Autoupdatemodelid.eval():
            self.auto_update_model_id_for_backend(backend)
        self.update_cn_id_menus()
        new_label = 'ControlNet [ Daydream: TOPin1 + Preprocessor ]' if backend == 'Daydream' else 'ControlNet [ Local: TOPin2 + Preprocessor ]'
        if self.ownerComp.par.Cnheader.label != new_label:
            self.ownerComp.par.Cnheader.label = new_label

        # Update Viewinstallguide label based on backend
        install_guide_label = 'Get API Key / Docs' if backend == 'Daydream' else 'Open Docs'
        if self.ownerComp.par.Viewinstallguide.label != install_guide_label:
            self.ownerComp.par.Viewinstallguide.label = install_guide_label

        self.check_install(force=True)

    def update_model_id_menu(self, backend):
        """Update Modelid menu options based on selected backend"""
        if backend == 'Daydream':
            # Only show Daydream-supported models
            self.ownerComp.par.Modelid.menuNames = DAYDREAM_SUPPORTED_MODELS
            self.ownerComp.par.Modelid.menuLabels = DAYDREAM_SUPPORTED_MODELS
            self.logger.log(f"Updated Model ID menu for Daydream backend: {len(DAYDREAM_SUPPORTED_MODELS)} models", level='DEBUG')
        else:
            self.sync_model_table()
            self.logger.log("Updated Model ID menu for Local backend (using model_table)", level='DEBUG')

    def auto_update_model_id_for_backend(self, backend):
        """Auto-update Modelid value when switching backends if needed"""
        current_model = self.ownerComp.par.Modelid.eval()
        if backend == 'Daydream':
            if current_model not in DAYDREAM_SUPPORTED_MODELS:
                self.ownerComp.par.Modelid = 'stabilityai/sd-turbo'
                self.logger.log(f"Auto-switched model from '{current_model}' to 'sd-turbo' (Daydream default)", level='INFO')
            else:
                self.logger.log(f"Model '{current_model}' is compatible with Daydream backend", level='DEBUG')

    def update_cn_id_menus(self):
        """
        Updates CN sequence block Id and Preprocessor parameter menus based on backend and model.
        Sets FULL HuggingFace IDs in menuNames (what gets sent to backend).
        Auto-updates preprocessor if Autopreprocess is enabled.
        Uses CN_MODEL_REGISTRY for consistent model mapping.
        """
        if not self.ownerComp.par.Autoupdatecn.eval():
            return

        backend = self.ownerComp.par.Backend.eval()
        current_model = self.ownerComp.par.Modelid.eval()

        # Determine base model type
        if backend == 'Daydream':
            # Map Daydream models to base types
            if current_model == "prompthero/openjourney-v4":
                base_model_type = 'sd15'
            elif current_model == "stabilityai/sd-turbo":
                base_model_type = 'sd21'
            elif current_model == "stabilityai/sdxl-turbo":
                base_model_type = 'sdxl'
            else:
                base_model_type = 'sdxl'  # Fallback
        else:
            # Local backend - detect from model
            config_type = self.get_config_type(current_model.lower())
            if config_type and config_type.startswith("sdxl"):
                base_model_type = 'sdxl'
            elif config_type == "sd21":
                base_model_type = 'sd21'
            else:
                base_model_type = 'sd15'

        # Get compatible models from registry
        menu_names = []
        menu_labels = []

        backend_key = backend.lower()  # 'daydream' or 'local'

        if base_model_type in CN_MODEL_REGISTRY:
            for model in CN_MODEL_REGISTRY[base_model_type]:
                # Filter by backend compatibility
                if backend_key in model['compatibility']:
                    menu_names.append(model['id'])
                    menu_labels.append(model['name'])

        # Fallback if no models found
        if not menu_names:
            menu_names = ['none']
            menu_labels = ['No Compatible Models']
            self.logger.log(f"No compatible CN models found for {base_model_type} on {backend} backend", level='WARNING')

        # Update each block in the CN sequence
        for block in self.ownerComp.par.Cn.sequence:
            block.par.Id.menuNames = menu_names
            block.par.Id.menuLabels = menu_labels

            # Auto-update preprocessor if Autopreprocess is enabled
            auto_preprocess_enabled = hasattr(self.ownerComp.par, 'Autopreprocess') and self.ownerComp.par.Autopreprocess.eval()

            if auto_preprocess_enabled:
                has_preprocessor = hasattr(block.par, 'Preprocessor')

                if has_preprocessor:
                    current_id = block.par.Id.eval()
                    detected_preprocessor = self.get_preprocessor_for_controlnet(current_id, backend)
                    block.par.Preprocessor.val = detected_preprocessor
        # Update dynamic CN parameters if Autoupdatecnpars is enabled
        self.update_cn_dynamic_parameters()

    def find_equivalent_cn_model(self, current_cn_id, target_base_model_type):
        """
        Find equivalent ControlNet model for different base model type.

        Args:
            current_cn_id: Current ControlNet model ID
            target_base_model_type: Target base model type ('sd15', 'sd21', 'sdxl')

        Returns:
            Equivalent CN model ID or None if no match found
        """
        # Extract CN type from current model
        current_cn_type = None
        current_base_type = None

        # Search through registry to find current model and its type
        for base_type, models in CN_MODEL_REGISTRY.items():
            for model in models:
                if model['id'] == current_cn_id:
                    current_cn_type = model['type']
                    current_base_type = base_type
                    break
            if current_cn_type:
                break

        if not current_cn_type:
            self.logger.log(f"Could not find CN type for '{current_cn_id}' in registry", level='DEBUG')
            return None

        # If already the right base type, no change needed
        if current_base_type == target_base_model_type:
            return current_cn_id

        # Find equivalent model in target base type
        if target_base_model_type in CN_MODEL_REGISTRY:
            for model in CN_MODEL_REGISTRY[target_base_model_type]:
                if model['type'] == current_cn_type:
                    return model['id']

        self.logger.log(f"No equivalent CN model found for type '{current_cn_type}' in '{target_base_model_type}'", level='DEBUG')
        return None

    def resolve_controlnet_model_id(self, model_id_input):
        """
        Maps short ControlNet names (like 'canny', 'depth') to full model IDs.
        Uses CN_MODEL_REGISTRY and current base model to find the correct full ID.

        Args:
            model_id_input: Either a short name ('canny') or full ID ('xinsir/controlnet-canny-sdxl-1.0')

        Returns:
            Full model ID string, or original input if no mapping found
        """
        # If already a full ID (contains '/'), return as-is
        if '/' in model_id_input:
            return model_id_input

        # If empty or 'none', return as-is
        if not model_id_input or model_id_input.lower() == 'none':
            return model_id_input

        # Determine current base model type from par.Modelid
        current_model = self.ownerComp.par.Modelid.eval()
        config_type = self.get_config_type(current_model.lower())

        if config_type and config_type.startswith("sdxl"):
            base_model_type = 'sdxl'
        elif config_type == "sd21":
            base_model_type = 'sd21'
        else:
            base_model_type = 'sd15'

        # Search registry for matching type
        short_name_lower = model_id_input.lower()

        if base_model_type in CN_MODEL_REGISTRY:
            for model in CN_MODEL_REGISTRY[base_model_type]:
                # Match by 'type' field (exact match, case-insensitive)
                if model['type'].lower() == short_name_lower:
                    return model['id']
                # Also try matching by 'name' field (e.g., "Canny", "Depth")
                if model['name'].lower() == short_name_lower:
                    return model['id']

        # No mapping found - return original input
        self.logger.log(f"Could not resolve ControlNet ID '{model_id_input}' for base model type '{base_model_type}'", level='WARNING')
        return model_id_input

    def auto_update_cn_values_for_model(self, model_id):
        """Auto-update ControlNet model selections when base model changes"""
        # Determine target base model type
        config_type = self.get_config_type(model_id.lower())

        if config_type and config_type.startswith("sdxl"):
            target_base_type = 'sdxl'
        elif config_type == "sd21":
            target_base_type = 'sd21'
        else:
            target_base_type = 'sd15'

        # Get backend for compatibility check
        backend = self.ownerComp.par.Backend.eval().lower()

        # Get all CN blocks
        if not hasattr(self.ownerComp.par, 'Cn'):
            return

        for block in self.ownerComp.par.Cn.sequence:
            try:
                current_cn_id = block.par.Id.eval()

                if not current_cn_id or current_cn_id == "none":
                    continue

                # Find equivalent CN model for new base model type
                new_cn_id = self.find_equivalent_cn_model(current_cn_id, target_base_type)

                if new_cn_id and new_cn_id != current_cn_id:
                    # Check compatibility with current backend
                    model_compatible = False
                    if target_base_type in CN_MODEL_REGISTRY:
                        for model in CN_MODEL_REGISTRY[target_base_type]:
                            if model['id'] == new_cn_id and backend in model['compatibility']:
                                model_compatible = True
                                break

                    if model_compatible:
                        block.par.Id.val = new_cn_id
                        self.logger.log(
                            f"Auto-updated CN from '{current_cn_id}' to '{new_cn_id}' for model '{model_id}'",
                            level='INFO'
                        )
                    else:
                        self.logger.log(
                            f"CN model '{new_cn_id}' not compatible with backend '{backend}'",
                            level='DEBUG'
                        )
            except Exception as e:
                self.logger.log(f"Error auto-updating CN block: {e}", level='ERROR')

    def Updatesettings(self):
        """
        Sends the four settings (Prompt, Delta, Guidance Scale, and Negative Prompt) to the OSC Out DAT.
        """
        if self.ownerComp.par.Streamactive:
            delta_value = self.ownerComp.par.Delta.eval()
            self.send_parameter_update('delta', delta_value)
            guidance_scale_value = self.ownerComp.par.Guidancescale.eval()
            self.send_parameter_update('guidance_scale', guidance_scale_value)
            self.Tindexblock()
            self.set_interpolation()
            if self.ownerComp.par.Usecontrolnet.eval():
                self.Cnblock()

    def send_osc_messages(self, message_dict):
        """
        Sends OSC messages to multiple addresses.

        Parameters:
        message_dict (dict): A dictionary where keys are OSC addresses and values are the messages to be sent.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')

            for address, message in message_dict.items():
                osc_out.sendOSC(address, [message])

    def send_parameter_update(self, param_name, value, update_type='single'):
        """
        Send parameter update via OSC to local Python process.
        Daydream backend is handled via parexec_daydream.py routing to submodule.

        Args:
            param_name (str): Parameter name (e.g. 'delta', 'guidance_scale')
            value: Parameter value
            update_type (str): 'single', 'json', 'list', 'command'
        """
        if not self.ownerComp.par.Streamactive and param_name != 'stop':
            return False

        backend = self.ownerComp.par.Backend.eval()

        if backend.lower() == 'local':
            # Local backend: send via OSC to local Python process
            return self._send_osc_parameter(param_name, value, update_type)

        # Daydream backend: parameters handled via parexec_daydream.py routing
        # which forwards to the daydream submodule
        return True

    def _send_osc_parameter(self, param_name, value, update_type):
        """Send parameter via OSC - simplified version matching working implementation"""
        try:
            osc_out = op('oscout1')
            if not osc_out:
                self.logger.log(f"ERROR: oscout1 not found for parameter {param_name}", level='ERROR')
                return False

            address = f'/{param_name}'
            # self.logger.log(f"Sending OSC: {address} = {str(value)[:100]} (type: {update_type})", level='INFO')

            if update_type == 'command':
                osc_out.sendOSC(address, [])
            elif update_type == 'json':
                import json
                osc_out.sendOSC(address, [json.dumps(value)])
            elif update_type == 'list':
                osc_out.sendOSC(address, value)
            else:  # single
                osc_out.sendOSC(address, [value])

            # self.logger.log(f"OSC sent successfully: {address}", level='INFO')
            return True
        except Exception as e:
            self.logger.log(f"OSC send failed for {param_name}: {str(e)}", level='ERROR')
            return False

    def _force_process_frame(self, reason):
        """
        Force a process frame command without state checks.
        Used for kickstarting feedback safe loops.
        
        Args:
            reason (str): Reason for forcing the frame (for logging)
        """
        if self.ownerComp.par.Serveractive and self.ownerComp.par.Streamactive:
            message_dict = {'/process_frame': 1}
            self.send_osc_messages(message_dict)
            self.logger.log(f'Forced process frame to {reason}', level='INFO')
        else:
            self.logger.log(f"Cannot force process frame ({reason}) - server/stream not active", level='WARNING')

    def Resolutionchange(self):
        """
        Called when Width or Height parameters change.
        Handles resolution validation and rounding based on acceleration mode.
        """
        # Guard against recursive calls when we set Width/Height inside round_width_height()
        if hasattr(self, '_in_resolution_change') and self._in_resolution_change:
            return

        try:
            self._in_resolution_change = True

            # Round width/height based on acceleration (TensorRT=64, other=8)
            needs_warning, rounding = self.round_width_height()

            # Get final values after rounding
            width = self.ownerComp.par.Width.eval()
            height = self.ownerComp.par.Height.eval()

            # Update stream name and reinit extensions
            date = datetime.datetime.now().strftime("%Y%m%d")
            new_name = f"stream_{width}x{height}_{date}"
            # self.ownerComp.par.Streamoutname = new_name
            # op('numpy_share_out').par.reinitextensions.pulse()
            # op('numpy_share_out_cn').par.reinitextensions.pulse()
            self.Streamoutname()

            # IMPORTANT: Final warning for dynamic expressions (must be last log)
            if needs_warning:
                backend = self.ownerComp.par.Backend.eval()
                if backend == 'Daydream':
                    backend_info = 'Daydream (TensorRT server-side)'
                else:
                    acceleration = self.ownerComp.par.Acceleration.eval()
                    backend_info = f'Local ({acceleration})'

                self.logger.log(
                    f"⚠️ NOTE: Ensure resolution {width}x{height} is divisible by {rounding} for {backend_info}. "
                    f"Width/Height use dynamic expressions - adjust your expression to round to {rounding}.",
                    level='WARNING'
                )
        finally:
            self._in_resolution_change = False


    def ImageChanged(self, info=None):
        """
        Callback triggered when StreamDiffusion output image changes.
        This is called from the shared memory extension when new frame content is detected.
        Much more efficient than TOP->CHOP conversion method.
        
        Auto-handles feedback safe mode when enabled.
        """
        try:
            current_timestamp = str(datetime.datetime.now())
            callback_data = {
                'timestamp': current_timestamp,
                'frame_source': 'shared_memory_detection',
                'detection_method': 'lightweight_hash'
            }
            
            # AUTO FEEDBACK SAFE MODE - handle frame-by-frame processing
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            is_stream_active = self.ownerComp.par.Streamactive.eval()
            is_manually_paused = self.ownerComp.par.Pausestream.eval()
            
            # Only trigger frame processing if:
            # - Feedback safe mode is enabled
            # - Stream is active  
            # - NOT manually paused (manual pause overrides feedback safe)
            if is_feedback_safe and is_stream_active and not is_manually_paused:
                # Auto-trigger single frame processing (automatic call)
                self.Processframe(is_manual=False)
                # self.logger.log("Auto-triggered feedback safe frame processing", level='DEBUG')
            
            # Handle normal image change callback
            if hasattr(self.ownerComp.par, 'Onimagechange') and self.ownerComp.par.Onimagechange:
                self.ownerComp.DoCallback("onImageChange", callback_data)
                
        except Exception as e:
            self.logger.log(f"Error in ImageChanged callback: {str(e)}", level='ERROR')


    def Promptblock(self):
        """
        Sends the concept and weight from each block in the Promptdict sequence to the OSC Out DAT.
        If par.Normpweights is True, the weights are normalized so that their sum equals the value of par.Totalpweights.
        Skips blocks with empty concept strings to prevent API errors.
        """
        prompt_list = []

        # Iterate over each block in the Promptdict sequence to build the list
        total_weight = 0
        for block in self.ownerComp.par.Promptdict.sequence:
            concept = block.par.Concept.eval()
            weight = block.par.Weight.eval()

            # Skip empty concept strings (prevents 500 error from API)
            if not concept or concept.strip() == '':
                continue

            prompt = [concept, weight]  # Create a list
            prompt_list.append(prompt)  # Append the list to the list
            if self.ownerComp.par.Normpweights:
                total_weight += weight  

        # Normalize the weights if required
        if self.ownerComp.par.Normpweights and prompt_list:
            if total_weight == 0:
                # Avoid division by zero - distribute weight equally across prompts
                equal_weight = self.ownerComp.par.Totalpweights / len(prompt_list)
                for i in range(len(prompt_list)):
                    prompt_list[i][1] = equal_weight
            else:
                normalization_factor = self.ownerComp.par.Totalpweights / total_weight
                for i in range(len(prompt_list)):
                    concept, weight = prompt_list[i]  # Unpack the list
                    weight *= normalization_factor
                    prompt_list[i] = [concept, weight]  # Create a new list

        # Convert the list to a JSON string
        prompt_list_str = json.dumps(prompt_list)
        # print(prompt_list_str)  

        # Send the JSON string via OSC
        if self.ownerComp.par.Streamactive:
            self.send_parameter_update('prompt_list', prompt_list_str, 'single')

    def Tindexblock(self):
        """
        Sends the t_index list with backend-specific parameter names.
        Local backend expects 't_list', Daydream expects 't_index_list'.
        """
        if self.ownerComp.par.Streamactive:
            t_index_list = []
            for block in self.ownerComp.par.Tindexblock.sequence:
                step_value = block.par.Step.eval()
                t_index_list.append(step_value)

            if len(t_index_list) != self.ownerComp.par.Tindexblock.sequence.numBlocks:
                return

            # Validate t_index_list is in non-decreasing order (required by Daydream API)
            for i in range(len(t_index_list) - 1):
                if t_index_list[i] > t_index_list[i + 1]:
                    self.logger.log(
                        f"WARNING: t_index_list must be in non-decreasing order. "
                        f"Found {t_index_list[i]} > {t_index_list[i + 1]} at position {i}. "
                        f"Current list: {t_index_list}",
                        level='WARNING'
                    )
                    break  # Only log once per send

            backend = self.ownerComp.par.Backend.eval()
            if backend.lower() == 'local':
                # Local backend expects 't_list'
                self.send_parameter_update('t_list', t_index_list, 'list')
            else:
                # Daydream backend expects 't_index_list'
                self.send_parameter_update('t_index_list', t_index_list, 'list')

    def Loradict(self):
        if self.ownerComp.par.Streamactive:
            import json
            lora_dict = {}
            if self.ownerComp.par.Uselora:
                for block in self.ownerComp.par.Loradictblock.sequence:
                    lora_path = block.par.Lorapath.eval()
                    if lora_path != 'select_lora_from_dropdown' and lora_path != '':
                        lora_dict[lora_path] = block.par.Weight.eval()
            self.send_parameter_update('lora_weights', json.dumps(lora_dict), 'single')

    def Hideuierrors(self):
        """Hide UI errors - Daydream errors handled by submodule"""
        self.logger.log("UI errors hidden", level='INFO')

    def Feedbacksafe(self, feedback_safe = None):
        # Handle feedback safe mode toggle - manages backend pause state for frame-by-frame control
        if feedback_safe is None:
            feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        
        # Only take action if the state has actually changed
        if feedback_safe != self.last_feedback_safe_state:
            if feedback_safe:
                # Enable feedback safe mode - put backend in pause mode for frame-by-frame control
                # Only pause if not already paused by user
                if not self.ownerComp.par.Pausestream.eval():
                    self.Pausestream(pause=True)
                    self.feedback_safe_paused_backend = True  # Remember we paused it
                    self.logger.log("Feedback safe mode enabled - backend paused for automatic frame-by-frame processing", level='INFO')
                    
                    # KICKSTART: Force initial frame to start the feedback safe loop (bypass state checks)
                    run('me.ext.StreamDiffusionExt._force_process_frame("start feedback safe loop")', delayFrames=1, fromOP=self.ownerComp)
                else:
                    self.feedback_safe_paused_backend = False  # User already paused
                    self.logger.log("Feedback safe mode enabled - using existing pause state", level='INFO')
            else:
                # Disable feedback safe mode - resume normal streaming if not manually paused
                is_manually_paused = self.ownerComp.par.Pausestream.eval()
                self.logger.log(f"DEBUG: Disabling feedback safe - feedback_safe_paused_backend={self.feedback_safe_paused_backend}, is_manually_paused={is_manually_paused}", level='DEBUG')
                
                # Always try to resume if not manually paused, regardless of who paused it initially
                if not is_manually_paused:
                    # User is not manually paused, so we should resume streaming
                    if self.ownerComp.par.Serveractive:
                        self.send_parameter_update('play', None, 'command')
                        self.logger.log("Feedback safe mode disabled - sent /play to resume normal streaming", level='INFO')
                        
                        # Also pulse the timers to ensure they don't timeout during transition
                        op('timer_stream').par.start.pulse()
                        op('timer_server').par.start.pulse()
                        self.logger.log("Pulsed timers to prevent timeout during feedback safe disable", level='DEBUG')
                    else:
                        self.logger.log("Server not active - cannot resume streaming", level='WARNING')
                else:
                    # User is manually paused, preserve that state
                    self.logger.log("Feedback safe mode disabled - manual pause state preserved", level='INFO')
                
                # Reset our tracking flag and record disable time
                self.feedback_safe_paused_backend = False
                self.last_feedback_safe_disable_time = time.time()
            
            # Update state tracking
            self.last_feedback_safe_state = feedback_safe

    def Seed(self): 
        """
        Sends the seed settings to the OSC Out DAT as a list of lists, combining weights for identical seeds before normalization.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            seed_weight_accumulator = {}
            for block in self.ownerComp.par.Seeddict.sequence:
                seed_val = block.par.Seedval.eval() 
                weight = block.par.Seedweight.eval()  
                if weight != 0:  
                    if seed_val in seed_weight_accumulator:
                        seed_weight_accumulator[seed_val] += weight  
                    else:
                        seed_weight_accumulator[seed_val] = weight  
            seed_list = [[seed, weight] for seed, weight in seed_weight_accumulator.items()]
            total_weight = sum(weight for _, weight in seed_list)
            if total_weight != 0:
                for i in range(len(seed_list)):
                    seed_val, weight = seed_list[i]
                    weight /= total_weight
                    seed_list[i] = [seed_val, weight]
            blend_multipliers = [1 + 0.5 * np.cos(np.pi * (weight - 0.5)) * (1 + 0.3 * (len(seed_list) - 2)) for _, weight in seed_list]
            avg_blend_multiplier = self.ownerComp.par.Noisemult.eval() * (sum(blend_multipliers) / len(blend_multipliers)) if blend_multipliers else 0
            for i in range(len(seed_list)):
                seed_val, weight = seed_list[i]
                weight *= avg_blend_multiplier
                seed_list[i] = [seed_val, weight]
            import json
            seed_list_str = json.dumps(seed_list)
            self.send_parameter_update('seed_list', seed_list_str, 'single')
            # self.logger.log(f'Sending seed update via OSC: {seed_list_str}', level='INFO')  # Add debug logging

    def Sdmode(self,sdmode = None):
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            if sdmode is None:
                sdmode = self.ownerComp.par.Sdmode.eval()
                if sdmode in ['txt2img', 'img2img']:
                    osc_out.sendOSC('/sdmode', [sdmode])

    def Limitfps(self):
        if self.ownerComp.par.Streamactive:
            max_fps_value = self.ownerComp.par.Limitfps.eval()
            self.send_parameter_update('max_fps', max_fps_value)

    def Usecontrolnet(self):
        if self.ownerComp.par.Streamactive:
            use_controlnet_value = self.ownerComp.par.Usecontrolnet.eval()

            if use_controlnet_value:
                self.Cnblock()
            else:
                # Unified disable path
                self.send_parameter_update('use_controlnet', False)
                self.ownerComp.storage['cn_use_controlnet_sent'] = False

    def Streamoutname(self):
        if self.ownerComp.par.Streamactive:
            stream_out_name_value = self.ownerComp.par.Streamoutname.eval()
            self.send_parameter_update('stream_out_name', stream_out_name_value)

    def Negprompt(self):
        if self.ownerComp.par.Streamactive:
            negative_prompt_message = self.ownerComp.par.Negprompt.eval()
            self.send_parameter_update('negative_prompt', negative_prompt_message)

    def set_interpolation(self):
        slerp = self.ownerComp.par.Setinterpolation.eval()
        if self.ownerComp.par.Streamactive:
            self.send_parameter_update('slerp', slerp)

    def round_width_height(self):
        """
        Rounds Width and Height parameters based on backend and acceleration.
        - Daydream: Always 64 (TensorRT server-side)
        - Local + TensorRT: 64 (latent space must be divisible by 8)
        - Local + other: 8 (standard VAE requirement)

        Only sets values if parameters are in CONSTANT mode (not expression/bind mode).
        Logs warnings for dynamic expressions to guide users.

        Returns: (needs_warning, rounding_value) tuple for use by caller
        """
        try:
            # Determine rounding based on backend and acceleration
            backend = self.ownerComp.par.Backend.eval()

            if backend == 'Daydream':
                # Daydream always uses TensorRT server-side
                rounding = 64
                backend_info = 'Daydream (TensorRT server-side)'
            else:
                # Local backend - check acceleration parameter
                acceleration = self.ownerComp.par.Acceleration.eval()
                rounding = 64 if acceleration == 'tensorrt' else 8
                backend_info = f'Local ({acceleration})'

            needs_warning = False

            # Handle Width parameter
            width_mode = self.ownerComp.par.Width.mode
            current_width = self.ownerComp.par.Width.eval()

            if width_mode == ParMode.CONSTANT:
                # User is directly setting value - round and set it back
                rounded_width = round(current_width / rounding) * rounding
                if rounded_width != current_width:
                    self.ownerComp.par.Width = rounded_width
            elif width_mode in [ParMode.EXPRESSION, ParMode.BIND]:
                # User has dynamic expression - don't modify, just check and warn
                needs_warning = True
                if current_width % rounding != 0:
                    self.logger.log(f"Width: Dynamic expression value {current_width} NOT divisible by {rounding} (required for {backend_info})", level='WARNING')

            # Handle Height parameter
            height_mode = self.ownerComp.par.Height.mode
            current_height = self.ownerComp.par.Height.eval()

            if height_mode == ParMode.CONSTANT:
                # User is directly setting value - round and set it back
                rounded_height = round(current_height / rounding) * rounding
                if rounded_height != current_height:
                    self.ownerComp.par.Height = rounded_height
            elif height_mode in [ParMode.EXPRESSION, ParMode.BIND]:
                # User has dynamic expression - don't modify, just check and warn
                needs_warning = True
                if current_height % rounding != 0:
                    self.logger.log(f"Height: Dynamic expression value {current_height} NOT divisible by {rounding} (required for {backend_info})", level='WARNING')

            return (needs_warning, rounding)

        except Exception as e:
            self.logger.log(f"Error rounding Width and Height: {e}", level='ERROR')
            return (False, 8)
    #endregion PARAMETER CALLBACKS

    #region STREAM CONTROL
    def Startstream(self):
        """
        Starts the StreamDiffusion stream.
        For Local backend: executes batch file to launch local Python process.
        For Daydream backend: handled via parexec_daydream.py routing to submodule.
        """
        if self.ownerComp.par.Serveractive:
            # Already running - send play command
            if self.ownerComp.par.Pausestream.mode == ParMode.CONSTANT:
                self.ownerComp.par.Pausestream = False
            self.send_parameter_update('play', None, 'command')
            self.logger.log('Playing stream...', level='INFO')
            return

        backend = self.ownerComp.par.Backend.eval()
        base_folder = self.ownerComp.par.Basefolder.eval()

        if backend == 'Daydream':
            # Daydream backend is handled via parexec_daydream.py routing to submodule
            # The submodule manages all WebRTC/API communication
            self.logger.log('Daydream backend - start handled by submodule', level='INFO')
            return

        # Local backend: launch local Python process
        # Re-activate timers if they were previously stopped
        for timer_name in ['timer_client', 'timer_server', 'timer_stream']:
            timer = op(timer_name)
            if timer:
                timer.par.active = 'always'
                self.logger.log(f"Re-activated {timer_name} for Local backend", level='DEBUG')

        self.copy_sdtd_code()

        # Check for virtual environment
        venv_path = os.path.join(base_folder, 'venv')
        dot_venv_path = os.path.join(base_folder, '.venv')
        if os.path.exists(venv_path):
            activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
        elif os.path.exists(dot_venv_path):
            activate_script_path = os.path.join(dot_venv_path, 'Scripts', 'activate.bat')
        else:
            self.logger.log("Check Install page / Installation not complete - Basefolder parameter must be set to '.../StreamDiffusion'", level="ERROR")
            return

        # Round width/height BEFORE generating YAML
        self.round_width_height()
        op('local_backend_status')['start_time', 1] = '0'
        self.copy_sdtd_code()

        python_script_path = 'streamdiffusionTD\\td_main.py'
        generation_command = f'{python_script_path}'
        log_message = 'Launching StreamDiffusionTD (Local)...'


        if not self.ownerComp.par.Visiblewindow:
            full_command = f'cmd.exe /c "{activate_script_path} && python {generation_command}"'
            try:
                subprocess.Popen(full_command, cwd=base_folder, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.logger.log(f'{log_message} (silently.)', level='INFO')
            except Exception as e:
                self.logger.log(f'Error Startstream: Failed to start streaming process without command window. Error: {e}', level='ERROR')
            return

        if platform.system() == 'Windows':
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.bat')
        elif platform.system() == 'Darwin':
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.sh')

        debug_cmd = 'pause' if self.ownerComp.par.Debugcmd.eval() else ''
        use_powershell = False
        if hasattr(self.ownerComp.par, 'Powershell'):
            use_powershell = self.ownerComp.par.Powershell.eval()
        
        # Only Local backend reaches here now (Daydream returns early)
        python_command_part = "'streamdiffusionTD\\td_main.py'"

        if use_powershell:
            # Get HF cache path if custom cache is enabled
            hf_cache_path = tdu.expandPath(self.ownerComp.par.Hfcache.eval()) if self.ownerComp.par.Sethfcache.eval() else ""
            hf_ps_env = f"$env:HF_HOME='{hf_cache_path}'; " if hf_cache_path else ""

            batch_file_content = f"""
@echo off
cd /d %~dp0
if exist venv (
    PowerShell -Command "& {{{hf_ps_env}& 'venv\\Scripts\\Activate.ps1'; & 'venv\\Scripts\\python.exe' {python_command_part}}}"
) else (
    PowerShell -Command "& {{{hf_ps_env}& '.venv\\Scripts\\Activate.ps1'; & '.venv\\Scripts\\python.exe' {python_command_part}}}"
)
    {debug_cmd}
            """
        else:
            # Get HF cache path if custom cache is enabled
            hf_cache_path = tdu.expandPath(self.ownerComp.par.Hfcache.eval()) if self.ownerComp.par.Sethfcache.eval() else ""
            hf_env_var = f"set HF_HOME={hf_cache_path}" if hf_cache_path else ""

            batch_file_content = f"""
            @echo off
            cd /d %~dp0
            {hf_env_var}
            if exist venv (
                call venv\\Scripts\\activate.bat
                venv\\Scripts\\python.exe {generation_command}
            ) else (
                call .venv\\Scripts\\activate.bat
                .venv\\Scripts\\python.exe {generation_command}
            )
            {debug_cmd}
            """
        if platform.system() == 'Darwin':
            mac_debug_cmd = 'read -p "Press any key to continue..." key' if self.ownerComp.par.Debugcmd.eval() else ''
            debug_at_start = False

            # Get HF cache path if custom cache is enabled (macOS)
            hf_cache_path = tdu.expandPath(self.ownerComp.par.Hfcache.eval()) if self.ownerComp.par.Sethfcache.eval() else ""
            hf_export = f"export HF_HOME='{hf_cache_path}'" if hf_cache_path else ""

            # Only Local backend reaches here now (Daydream returns early)
            mac_python_command = "python streamdiffusionTD/td_main.py"

            batch_file_content = f"""
                #!/bin/sh
                # Unset PYTHONPATH to avoid TD Python interference
                unset PYTHONPATH

                # Set HuggingFace cache location if custom path specified
                {hf_export}

                cd "$(dirname "$0")"
                
                if {debug_at_start}; then
                    echo "Current directory: $(pwd)"
                    echo "PATH: $PATH"
                    echo "Available Python versions:"
                    which -a python python3
                fi
                
                if [ -d "venv" ]; then
                    source venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using venv at: $(which python)"
                        echo "Python version: $(python --version)"
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                elif [ -d ".venv" ]; then
                    source .venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using .venv at: $(which python)"
                        echo "Python version: $(python --version)"
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                else
                    source daydream_venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using daydream_venv at: $(which python)"
                        echo "Python version: $(python --version)" 
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                fi
                {mac_debug_cmd}
                """
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)
        if platform.system() == 'Windows':
            creation_flags = self.get_subprocess_creationflags()
            if creation_flags:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval(), creationflags=creation_flags)
            else:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
            self.logger.log(f'{log_message}', level='INFO')
        elif platform.system() == 'Darwin':
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
            self.logger.log(f'{log_message}', level='INFO')

    def Stopstream(self):
        """
        Stops the StreamDiffusion stream.
        For Local backend: sends /stop OSC command.
        For Daydream backend: handled via parexec_daydream.py routing to submodule.
        """
        if not self.ownerComp.par.Serveractive:
            return

        self.send_parameter_update('stop', None, 'command')
        self.logger.log('Stopping server...', level='INFO')
        run("if me.par.Streamactive.mode == ParMode.CONSTANT: me.par.Streamactive = False", delayFrames=2, fromOP=self.ownerComp)
        run("if me.par.Pausestream.mode == ParMode.CONSTANT: me.par.Pausestream = False", delayFrames=3, fromOP=self.ownerComp)
        self.last_stop_time = time.time()

        backend = self.ownerComp.par.Backend.eval()

        if backend == 'Daydream':
            # Daydream backend shutdown is handled via parexec_daydream.py routing to submodule
            self.logger.log('Daydream backend - stop handled by submodule', level='INFO')
            if self.ownerComp.par.Serveractive.mode == ParMode.CONSTANT:
                self.ownerComp.par.Serveractive = False
        else:
            # Local backend: stop webrender
            op('webrender_daydream').par.active = False

    def Pausestream(self, pause=None):
        """
        Sends a /pause or /play command to the OSC server based on the Pausestream parameter.
        Can accept either a TouchDesigner parameter object or a boolean value.
        Handles feedback safe mode properly.
        """
        if pause is None:
            pause = self.ownerComp.par.Pausestream
        if self.ownerComp.par.Serveractive:
            osc_out = op('oscout1')
            # Handle both parameter objects and boolean values
            pause_value = pause.eval() if hasattr(pause, 'eval') else pause
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            
            if pause_value:
                # Manual pause - always pause backend
                self.send_parameter_update('pause', None, 'command')
                self.logger.log('Pausing stream...', level='INFO')
            else:
                # Manual unpause - check if we're in feedback safe mode
                if is_feedback_safe:
                    # In feedback safe mode - keep backend paused for frame-by-frame control
                    self.send_parameter_update('pause', None, 'command')
                    self.logger.log('Resuming stream in feedback safe mode (backend stays paused for frame-by-frame control)...', level='INFO')
                    
                    # KICKSTART: Force initial frame to restart the feedback safe loop (bypass state checks)
                    run('me.ext.StreamDiffusionExt._force_process_frame("restart feedback safe loop")', delayFrames=1, fromOP=self.ownerComp)
                else:
                    # Normal mode - resume streaming
                    self.send_parameter_update('play', None, 'command')
                    self.logger.log('Resuming stream...', level='INFO')
        else:
            if self.ownerComp.par.Pausestream.mode == ParMode.CONSTANT: 
                self.ownerComp.par.Pausestream = False

    def Serveractive(self):
        self.Promptblock()
        if self.ownerComp.par.Serveractive:
            if self.ownerComp.par.Startstream.label != "Start Stream":
                self.ownerComp.par.Startstream.label = "Start Stream"
            if self.ownerComp.par.Stopstream.label != "Stop Server":
                self.ownerComp.par.Stopstream.label = "Stop Server"
            op('timer_server').par.start.pulse()
            if not self.ownerComp.par.Streamactive:
                if self.ownerComp.par.Acceleration == 'tensorrt':
                    self.logger.log('Server active... Loading TensorRT Models...', level='INFO')
                else:
                    self.logger.log('Server active... Loading Models...', level='INFO')
                return
        else:
            if self.ownerComp.par.Startstream.label != "Start Stream":
                self.ownerComp.par.Startstream.label = "Start Stream"
            if self.ownerComp.par.Stopstream.label != "Stop Server":
                self.ownerComp.par.Stopstream.label = "Stop Server"
            self.logger.log('■ Server stopped...', level='INFO')

            backend = self.ownerComp.par.Backend.eval()
            if backend == 'Daydream':
                # Daydream status handled by submodule
                pass
            elif backend == 'local':
                # Clear local_backend_status streaming fields
                local_status = op('local_backend_status')
                if local_status:
                    # Only set server_stopped if not in a loading state
                    current_state = str(local_status['connection_state', 1].val)
                    if current_state not in ['local_initializing', 'local_loading_models']:
                        self._clear_local_status_table(local_status)
                        local_status['connection_state', 1] = 'server_stopped'

    def Processframe(self, is_manual=True):
        """
        Triggers processing of a single frame when in pause mode.
        Used for manual frame-by-frame control and loopback scenarios.
        
        Args:
            is_manual (bool): True if called manually (button/pulse), False if called automatically (feedback safe)
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_processframe_time < self.processframe_debounce_ms:
            self.logger.log(f"Processframe debounced (last call {current_time - self.last_processframe_time:.1f}ms ago)", level='DEBUG')
            return
        self.last_processframe_time = current_time        
        delay_frames = self.ownerComp.par.Delayframes.eval()
        run(f'me.ext.StreamDiffusionExt.send_processframe({is_manual})',fromOP=self.ownerComp,delayFrames=delay_frames)


    def send_processframe(self, is_manual=True):
        """
        Triggers processing of a single frame when in pause mode.
        Used for manual frame-by-frame control and loopback scenarios.
        
        Args:
            is_manual (bool): True if called manually, False if called automatically
        """

        
        # Check current state
        is_paused = self.ownerComp.par.Pausestream.eval()
        is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        is_stream_active = self.ownerComp.par.Streamactive.eval()
        
        # Determine if we should process based on manual vs automatic call
        should_process = False
        
        if is_stream_active:
            if is_paused and is_feedback_safe:
                # Manually paused + feedback safe: only allow manual process frame calls
                should_process = is_manual
                if not is_manual:
                    self.logger.log("Auto process frame ignored - manually paused (use manual process frame)", level='DEBUG')
            elif is_paused:
                # Only manually paused: allow manual calls
                should_process = is_manual
            elif is_feedback_safe:
                # Only feedback safe (not manually paused): allow automatic calls
                should_process = not is_manual
            else:
                # Neither paused nor feedback safe: check if we're in transition grace period
                time_since_disable = time.time() - self.last_feedback_safe_disable_time
                grace_period = 2.0  # 2 second grace period for queued automatic calls
                
                if not is_manual and time_since_disable < grace_period:
                    # Allow automatic calls during grace period (queued from when feedback safe was enabled)
                    should_process = False  # Still don't process, but don't log as error
                    self.logger.log(f"Auto process frame ignored - feedback safe recently disabled ({time_since_disable:.1f}s ago)", level='DEBUG')
                else:
                    # Normal case: shouldn't be calling process frame
                    should_process = False
        
        if should_process:
            current_timestamp = str(datetime.datetime.now())
            message_dict = {'/process_frame': 1}
            self.send_osc_messages(message_dict)
            
            # Keep timers alive during feedback safe mode
            if is_feedback_safe:
                op('timer_stream').par.start.pulse()
                op('timer_server').par.start.pulse()
            
            call_type = "manual" if is_manual else "automatic"
            # self.logger.log(f"Triggered {call_type} single frame processing", level='DEBUG')
        else:
            call_type = "manual" if is_manual else "automatic"
            self.logger.log(f"Process frame ignored - {call_type} call not appropriate for current state (paused={is_paused}, feedback_safe={is_feedback_safe})", level='DEBUG')

    def onFrameReady(self, frame_count):
        """
        Called when server signals that a frame is ready.
        Send acknowledgment immediately for proper loopback synchronization.
        """
        try:
            current_timestamp = str(datetime.datetime.now())
            callback_data = {
                'timestamp': current_timestamp,
                'frame_count': frame_count,
                'sync_type': 'server_frame_ready'
            }
            
            # self.logger.log(f"✓ RECEIVED /frame_ready signal from server for frame {frame_count}", level='INFO')
            
            # LOOPBACK SYNCHRONIZATION: Send frame acknowledgment when in pause mode, feedback safe mode, 
            # OR during grace period after feedback safe is disabled (backend still expects acknowledgments)
            is_paused = self.ownerComp.par.Pausestream.eval()
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            
            # Check if we're in the grace period after feedback safe was disabled
            time_since_disable = time.time() - self.last_feedback_safe_disable_time
            grace_period = 5.0  # Match backend's 5 second acknowledgment timeout
            in_feedback_safe_grace_period = not is_feedback_safe and time_since_disable < grace_period
            
            if is_paused or is_feedback_safe or in_feedback_safe_grace_period:
                success = self.send_frame_acknowledgment()
                if not success:
                    self.logger.log("FAILED to send frame acknowledgment!", level='ERROR')
                elif in_feedback_safe_grace_period:
                    self.logger.log(f"Sent frame acknowledgment during feedback safe grace period ({time_since_disable:.1f}s ago)", level='DEBUG')
            
            # Call frame ready callback if enabled
            if hasattr(self.ownerComp.par, 'Onframeready') and self.ownerComp.par.Onframeready:
                self.ownerComp.DoCallback("onFrameReady", callback_data)
                
        except Exception as e:
            self.logger.log(f"Error in onFrameReady callback: {str(e)}", level='ERROR')

    def Streamactive(self):
        if not self.ownerComp.par.Serveractive:
            self.ownerComp.par.Streamactive = False 
        else:
            if self.ownerComp.par.Streamactive:
                op('timer_stream').par.start.pulse()
        current_timestamp = str(datetime.datetime.now())
        callback_data = {
            'timestamp': current_timestamp,
        }
        if self.ownerComp.par.Streamactive:
            self.Updatestreamname()
            self.Updatesettings()
            self.Promptblock()
            self.Seed()  # Add missing seed synchronization on stream start
            self.Loradict()  # Also add LoRA sync for completeness  
            run("me.Usecontrolnet()", fromOP = self.ownerComp, delayFrames = 10)
            run("me.Updatesettings()", fromOP = self.ownerComp, delayFrames = 10)
            run("me.Seed()", fromOP = self.ownerComp, delayFrames = 15)  # Add delayed seed sync as backup
            # Re-sync ControlNet after backend initialization settles.
            run("if me.par.Streamactive and me.par.Usecontrolnet: me.Cnblock()", fromOP=self.ownerComp, delayFrames=30)
            run("if me.par.Streamactive and me.par.Usecontrolnet: me.Cnblock()", fromOP=self.ownerComp, delayFrames=60)
            # op('numpy_share_out').par.reinitextensions.pulse()
            # op('numpy_share_out_cn').par.reinitextensions.pulse()
            if self.last_stop_time and time.time() - self.last_stop_time > 1:
                if self.ownerComp.par.Pausestream.eval():
                    return
                if self.ownerComp.par.Onstreamstart:
                    self.ownerComp.DoCallback("onStreamStart", callback_data)
                self.logger.log("onStreamStart: StreamDiffusion has begun streaming", f"Called onStreamStart with data: {callback_data}", level='INFO')
        else:
            pause = self.ownerComp.par.Pausestream.eval() #bool
            server = self.ownerComp.par.Serveractive.eval() #bool

    def onStreamEnd(self, pause, server):
        current_timestamp = str(datetime.datetime.now())
        callback_data = {
            'timestamp': current_timestamp,
            'is_paused': pause,
            'server_status': server,
        }
        
        # FEEDBACK SAFE MODE: If stream ends but we're in feedback safe mode, try to restart processing
        is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        if is_feedback_safe and server and not pause:
            self.logger.log("onStreamEnd: Stream ended in feedback safe mode - attempting to restart frame processing", level='INFO')
            # Try to trigger another frame to keep the feedback loop going
            try:
                self.Processframe(is_manual=False)  # Automatic restart
                self.logger.log("Successfully triggered restart frame in feedback safe mode", level='INFO')
                return  # Don't call the normal onStreamEnd callback - we're restarting
            except Exception as e:
                self.logger.log(f"Failed to restart feedback safe processing: {str(e)}", level='WARNING')
                # Fall through to normal stream end handling
        
        # Normal stream end handling
        if self.ownerComp.par.Onstreamend:
            self.ownerComp.DoCallback("onStreamEnd", callback_data) 
            if pause and server:
                self.logger.log("onStreamEnd: PAUSED.", f"Called onStreamEnd with data: {callback_data}", level='INFO')
            elif not server:
                self.logger.log("onStreamEnd: SERVER SHUTDOWN.", f"Called onStreamEnd with data: {callback_data}", level='INFO')
            else:
                self.logger.log("onStreamEnd: StreamDiffusion has ended streaming.", f"Called onStreamEnd with data: {callback_data}", level='INFO')

    def onServerEnd(self):
        self.logger.log("The server is shut down.", f"logged from onServerEnd", level='INFO')

    # def Enablefx(self):
    #     osc_out = op('oscout1')
    #     self.send_parameter_update('disable_cached_attn', 1-self.ownerComp.par.Enablefx.eval(), 'single')

    def Synccompcreate(self):
        """
        Duplicates the '_content' operator inside self.ownerComp to the parent level and positions it to the left and up a bit.
        """
        content_op = self.ownerComp.op('_content')
        if content_op:
            duplicated_op = self.ownerComp.parent().copy(content_op)
            duplicated_op.nodeX = self.ownerComp.nodeX - 200  # Move left by 200 units
            duplicated_op.nodeY = self.ownerComp.nodeY + 125  # Move up by 125 units
            duplicated_op.allowCooking = 1 
            new_name = duplicated_op.name
            self.logger.log('StreamDiffusionExt: SyncComp created',
                            f'Operator {new_name} duplicated to parent level and repositioned.',
                            level='INFO')
            callback_cmd = f"\tif op('{new_name}'): op('{new_name}').par.Play = 1; run(\"op('{new_name}').par.Play = 0\", delayFrames=1, fromOP=me)"
            callbackdat = op(self.ownerComp.par.Callbackdat)
            current_callback_code = callbackdat.text
            insert_point = current_callback_code.find('def onReceiveFrame(info):') + len('def onReceiveFrame(info):')
            if callback_cmd not in current_callback_code:
                new_callback_code = (current_callback_code[:insert_point] + 
                                     '\n' + callback_cmd + 
                                     current_callback_code[insert_point:])
                callbackdat.text = new_callback_code
                self.logger.log('StreamDiffusionExt: Callback updated',
                                f'Callback command inserted correctly for operator {new_name}.',
                                level='INFO')
            else:
                self.logger.log('StreamDiffusionExt: Callback already exists',
                                f'Callback command for operator {new_name} is already present.',
                                level='INFO')
        else:
            self.logger.log('StreamDiffusionExt: SyncComp creation failed',
                            'No operator named "_content" found inside the component.',
                            level='ERROR')            

    def Updatestreamname(self):
        try:
            source_name = op('stream_osc_data')[1, 'output-name'].val.lower()
            current_Streaminname = self.ownerComp.par.Streaminname.eval()
            # Check if any of the menu options in self.ownerComp.par.Streaminname contain the source_name
            for i, menu_label in enumerate(self.ownerComp.par.Streaminname.menuLabels):
                if source_name in menu_label.lower():
                    # If found, set the par.Streaminname to that
                    self.ownerComp.par.Streaminname = self.ownerComp.par.Streaminname.menuNames[i]
                    break
            # If par.Streaminname has changed, log the change
            if self.ownerComp.par.Streaminname.eval() != current_Streaminname:
                self.logger.log(f'Source Detected. Source name changed to {self.ownerComp.par.Streaminname.eval()}', level='INFO')
        except:
            pass
    #endregion STREAM CONTROL

    #region SETUP AND CLONE
    def Clonestreamdiffusion(self):
        if self.message_box_open:
            return
        self.message_box_open = True
        repo_url = 'https://github.com/kramiusmaximus/StreamDiffusion.git'
        base_folder_param = self.ownerComp.par.Basefolder
        chosen_folder = base_folder_param.eval() if base_folder_param and base_folder_param.eval() else None

        # If no folder chosen yet, ask for one
        if not chosen_folder:
            chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
            if not chosen_folder:
                self.message_box_open = False
                return False

        # Check for spaces BEFORE doing anything else
        while ' ' in chosen_folder:
            choice = ui.messageBox('Installation Path Error',
                                f'The selected path contains spaces which will cause errors:\n\n' +
                                f'Selected path: {chosen_folder}\n\n' +
                                'Would you like to choose a different location?',
                                buttons=['Choose New Location', 'Cancel'])
            if choice == 0:  # Choose New Location
                chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository (No Spaces)')
                if not chosen_folder:
                    self.message_box_open = False
                    return False
            else:  # They hit Cancel
                self.message_box_open = False
                return False

        # NOW we can check if it's a git repo and proceed with cloning
        if os.path.isdir(os.path.join(chosen_folder, '.git')):
            # If it is, ask the user what they want to do
            choice = ui.messageBox('Git Repository Detected',
                                "The selected folder is already a Git repository.\n"
                                "What would you like to do?",
                                buttons=['Update Repo', 'Choose New Download Location', 'Cancel'])
            if choice == 0:  # Update Repo
                try:
                    # First, check current branch
                    check_branch_cmd = ['git', '-C', chosen_folder, 'rev-parse', '--abbrev-ref', 'HEAD']
                    branch_process = subprocess.Popen(check_branch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    current_branch, _ = branch_process.communicate()
                    current_branch = current_branch.strip()

                    # If not on SDTD_031_stable, switch to it
                    if current_branch != 'SDTD_031_stable':
                        self.logger.log(f'Current branch is {current_branch}, switching to SDTD_031_stable...', level='INFO')
                        # Fetch all branches first
                        fetch_cmd = ['git', '-C', chosen_folder, 'fetch', 'origin']
                        subprocess.run(fetch_cmd, check=True, capture_output=True)

                        # Checkout SDTD_031_stable
                        checkout_cmd = ['git', '-C', chosen_folder, 'checkout', 'SDTD_031_stable']
                        checkout_process = subprocess.Popen(checkout_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        _, checkout_err = checkout_process.communicate()
                        if checkout_process.returncode != 0:
                            ui.messageBox('Error', f'Error switching to SDTD_031_stable branch:\n{checkout_err}')
                            self.logger.log(f'Install 1:  Error switching to SDTD_031_stable branch:\n{checkout_err}', level='ERROR')
                            self.message_box_open = False
                            return False

                    # Now pull latest changes from SDTD_031_stable
                    command = ['git', '-C', chosen_folder, 'pull', 'origin', 'SDTD_031_stable']
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        ui.messageBox('Error', f'Error updating repository:\n{stderr}')
                        self.logger.log(f'Install 1:  Error updating repository:\n{stderr}', level='ERROR')
                        self.message_box_open = False
                        return False
                    else:
                        self.logger.log('Update successful - now on SDTD_031_stable branch with latest changes.', level='INFO')
                        # Also ensure installer repo is present/updated
                        self.clone_installer_repo()
                        self.message_box_open = False
                        return True
                except Exception as e:
                    ui.messageBox('Error', f'Failed to execute the command: {e}')
                    self.logger.log(f'Install 1:  Failed to execute the command: {e}', level='ERROR')
                    self.message_box_open = False
                    return False
            elif choice == 1:  # Pick New Location for Fresh Download
                chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
                if chosen_folder is None:
                    self.message_box_open = False
                    return False
                # Check new location for spaces
                while ' ' in chosen_folder:
                    choice = ui.messageBox('Installation Path Error',
                                        f'The selected path contains spaces which will cause errors:\n\n' +
                                        f'Selected path: {chosen_folder}\n\n' +
                                        'Would you like to choose a different location?',
                                        buttons=['Choose New Location', 'Cancel'])
                    if choice == 0:  # Choose New Location
                        chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository (No Spaces)')
                        if not chosen_folder:
                            self.message_box_open = False
                            return False
                    else:  # They hit Cancel
                        self.message_box_open = False
                        return False
            else:  # Cancel or 'x' button
                self.message_box_open = False
                return False

        # If it's not a git repository, or if the user chose to download to a new location, proceed with cloning
        success = self.clone_git_to_folder(repo_url, chosen_folder=chosen_folder, folder_parameter='Basefolder')
        if success:
            self.logger.log('StreamDiffusion clone successful.', level='INFO')
            # Clone the installer into the base folder
            if self.clone_installer_repo():
                self.logger.log('Installer clone successful. Ready for Step 2.', level='INFO')
            else:
                self.logger.log('Installer clone failed. You can retry or clone manually.', level='WARNING')
        else:
            self.logger.log('StreamDiffusion clone failed.', level='ERROR')
        self.message_box_open = False

    def clone_git_to_folder(self, repo_url, chosen_folder=None, folder_parameter=None, branch='SDTD_031_stable'):
        if not self.is_git_installed():
            ui.messageBox('Git Not Found', 'Git is not installed on this system. Please install Git.')
            self.logger.log('Install 1:  Git Not Found', level='ERROR')
            return False
        if chosen_folder is None:
            chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
        if chosen_folder is None:
            return False
        git_folder_name = repo_url.split('/')[-1].replace('.git', '')
        clone_destination = os.path.join(chosen_folder, git_folder_name)
        if (platform.system() == 'Windows'):
            clone_destination = clone_destination.replace('/', '\\')  # Ensure the path uses backslashes
        try:
            command = ['git', 'clone', '--branch', branch, repo_url, clone_destination]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                ui.messageBox('Error', f'Error cloning repository:\n{stderr}')
                self.logger.log(f'Install 1:  Error cloning repository:\n{stderr}', level='ERROR')
                return False
        except Exception as e:
            ui.messageBox('Error', f'Failed to execute the command: {e}')
            self.logger.log(f'Install 1:  Failed to execute the command: {e}', level='ERROR')
            return False
        if folder_parameter:
            try:
                setattr(self.ownerComp.par, folder_parameter, clone_destination)
            except Exception as e:
                ui.messageBox('Error', f'Failed to set the parameter: {folder_parameter}')
                self.logger.log(f'Install 1:  Failed to set the parameter: {folder_parameter}', level='ERROR')
                return False
        return True

    def clone_installer_repo(self):
        """Clone or update StreamDiffusion-installer into par.Basefolder."""
        base_folder = self.ownerComp.par.Basefolder.eval()
        if not base_folder:
            self.logger.log('Cannot clone installer: Basefolder not set', level='ERROR')
            return False

        installer_url = 'https://github.com/dotsimulate/StreamDiffusion-installer.git'
        installer_dest = os.path.join(base_folder, 'StreamDiffusion-installer')
        if platform.system() == 'Windows':
            installer_dest = installer_dest.replace('/', '\\')

        if os.path.isdir(os.path.join(installer_dest, '.git')):
            self.logger.log('Updating StreamDiffusion-installer...', level='INFO')
            try:
                cmd = ['git', '-C', installer_dest, 'pull', 'origin', 'main']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                _, stderr = process.communicate()
                if process.returncode != 0:
                    self.logger.log(f'Installer update failed: {stderr}', level='WARNING')
                    return False
                return True
            except Exception as e:
                self.logger.log(f'Installer update error: {e}', level='WARNING')
                return False
        else:
            self.logger.log('Cloning StreamDiffusion-installer...', level='INFO')
            try:
                cmd = ['git', 'clone', installer_url, installer_dest]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                _, stderr = process.communicate()
                if process.returncode != 0:
                    self.logger.log(f'Installer clone failed: {stderr}', level='ERROR')
                    return False
                return True
            except Exception as e:
                self.logger.log(f'Installer clone error: {e}', level='ERROR')
                return False

    def is_git_installed(self):
        try:
            subprocess.check_output(["git", "--version"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_tensorrt_engine_compatibility(self):
        """
        Checks if the current model specifications and parameter settings
        match an existing TensorRT engine.
        
        Returns:
        bool: True if a compatible engine exists, False otherwise.
        str: A message explaining the result.
        """
        if self.ownerComp.par.Acceleration.eval() != 'tensorrt':
            return False, "TensorRT is not selected as the acceleration method."

        engines_folder = self.ownerComp.par.Enginefolder.eval()
        if not os.path.exists(engines_folder):
            return False, "No engines folder found. TensorRT engines need to be built."

        current_config = self.get_current_engine_config()
        engine_pattern = self.generate_engine_pattern(current_config)

        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                if engine_pattern.search(subdir):
                    engine_path = os.path.join(root, subdir)
                    required_files = ['unet.engine', 'vae_encoder.engine', 'vae_decoder.engine']
                    missing_files = [file for file in required_files if file not in os.listdir(engine_path)]
                    
                    if not missing_files:
                        return True, f"Compatible TensorRT engine found: {subdir}"
                    else:
                        return False, f"Incomplete engine found: {subdir}. Missing files: {', '.join(missing_files)}"

        return False, "No compatible TensorRT engine found. A new engine may need to be built, which could take a long time."

    def get_current_engine_config(self):
        return {
            'model_id': self.normalize_model_id(self.ownerComp.par.Modelid.eval()),
            'lcm_lora': self.normalize_model_id(self.ownerComp.par.Customlcm.eval()) if self.ownerComp.par.Usecustomlcm.eval() else None,
            'max_batch': self.ownerComp.par.Tindexblock.sequence.numBlocks,
            'width': self.ownerComp.par.Width.eval(),
            'height': self.ownerComp.par.Height.eval()
        }

    def generate_engine_pattern(self, config):
        """
        Generates a regex pattern to match compatible TensorRT engines.
        New LivePeer fork engines are resolution-independent, so we only match on:
        - Model name (base_name from normalized model_id)
        - Batch size (max_batch)
        - Mode (img2img/txt2img)
        - Optional: FaceID flag (--fid)
        """
        # Extract model base name from model_id (e.g., "stabilityai/sdxl-turbo" -> "sdxl-turbo")
        model_parts = config['model_id'].split('/')
        base_name = model_parts[-1] if len(model_parts) > 1 else config['model_id']

        # Check if IPAdapter FaceID is enabled
        is_faceid = hasattr(self.ownerComp.par, 'Ipfaceid') and self.ownerComp.par.Ipfaceid.eval()

        # Build pattern: base_name--lcm_lora-*--tiny_vae-*--min_batch-*--max_batch-N--mode-*
        # Optional --fid flag for FaceID engines
        # Resolution-independent: no width/height in pattern!
        max_batch = config['max_batch']

        if is_faceid:
            # Match engines with --fid flag and matching max_batch
            pattern_str = rf'{re.escape(base_name)}--.*--max_batch-{max_batch}--.*--fid--.*--mode-'
        else:
            # Match engines without --fid flag and matching max_batch
            # Negative lookahead to exclude --fid engines
            pattern_str = rf'{re.escape(base_name)}--.*--max_batch-{max_batch}--(?!.*--fid).*--mode-'

        return re.compile(pattern_str)

    def update_local_model_par(self, was_cloned, target_folder, parameter_name):
        """
        Updates the component parameters if the model is downloaded successfully.
        """
        if was_cloned:
            use_parameter_name = 'Use' + parameter_name.lower()
            setattr(self.ownerComp.par, use_parameter_name, True)
            setattr(self.ownerComp.par, parameter_name, target_folder)
            self.logger.log(f'Parameters updated for {parameter_name} with {target_folder}', level='INFO')

    def get_venv_python(self):
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')
        
        if os.path.exists(venv_path):
            if platform.system() == 'Windows':
                return os.path.join(venv_path, 'Scripts', 'python.exe')
            elif platform.system() == 'Darwin':  # macOS
                return os.path.join(venv_path, 'bin', 'python')
            else:
                self.logger.log("Unsupported operating system.", level="ERROR")
                return None
        else:
            self.logger.log("Virtual environment not found.", level="ERROR")
            return None

    def run_script_in_venv(self, script):
        venv_python = self.get_venv_python()
        if not venv_python:
            return None
        try:
            if platform.system() == 'Windows':
                creation_flags = subprocess.CREATE_NO_WINDOW
            else:
                creation_flags = 0
            process = subprocess.Popen(
                [venv_python, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=creation_flags
            )
            stdout, stderr = process.communicate()
            if stderr:
                self.logger.log(f"Error running script: {stderr}", level="ERROR")
                return None
            return stdout
        except Exception as e:
            self.logger.log(f"Error running script: {str(e)}", level="ERROR")
            return None

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                self.logger.log(f"Created folder: {folder_path}", level="INFO")
            except OSError as e:
                self.logger.log(f"Error creating folder: {folder_path}. Error: {e}", level="ERROR")
    #endregion SETUP AND CLONE

    #region UI HELPERS
    def Uiviewlora(self):
        self.ownerComp.par.Uiviewlora = 0
        if time.time() - self.last_opened_time > 1:
            lora_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LoRA')
            self.create_folder_if_not_exists(lora_folder)
            ui.viewFile(lora_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewlora = 0

    def Uiviewmodels(self):
        self.ownerComp.par.Uiviewmodels = 0
        if time.time() - self.last_opened_time > 1:
            model_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models')
            self.create_folder_if_not_exists(model_folder)
            ui.viewFile(model_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewmodels = 0

    def Uiviewlcm(self):
        self.ownerComp.par.Uiviewlcm = 0
        if time.time() - self.last_opened_time > 1:
            lcm_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LCM')
            self.create_folder_if_not_exists(lcm_folder)
            ui.viewFile(lcm_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewlcm = 0

    def Uiviewvae(self):
        self.ownerComp.par.Uiviewvae = 0
        if time.time() - self.last_opened_time > 1:
            vae_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'VAE')
            self.create_folder_if_not_exists(vae_folder)
            ui.viewFile(vae_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewvae = 0
    #endregion UI HELPERS

    #region INSTALLATION
    def find_python_exe(self):
        if platform.system() == 'Darwin':  # macOS specific handling
            # First try Homebrew Python 3.11 specifically
            homebrew_paths = [
                "/opt/homebrew/bin/python3.11",
                "/usr/local/opt/python@3.11/bin/python3.11",
                "/opt/homebrew/opt/python@3.11/bin/python3.11"
            ]
            
            # Try to find Homebrew Python 3.11 first
            for path in homebrew_paths:
                if os.path.exists(path):
                    try:
                        output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match and version_match.group(1).startswith('3.11'):
                            return path, version_match.group(1)
                    except:
                        continue

            # If Homebrew Python 3.11 not found, try to find it using brew
            try:
                brew_path = subprocess.check_output(['which', 'brew']).decode().strip()
                if brew_path:
                    python_path = subprocess.check_output([brew_path, '--prefix', 'python@3.11']).decode().strip()
                    python_exe = os.path.join(python_path, 'bin', 'python3.11')
                    if os.path.exists(python_exe):
                        output = subprocess.check_output([python_exe, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            return python_exe, version_match.group(1)
            except:
                pass

        else:  # Windows paths
            # Method 1: Try LOCALAPPDATA environment variable first (most reliable)
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                # First check Python.org installations (priority)
                python_paths = [
                    os.path.join(localappdata, 'Programs', 'Python', 'Python311', 'python.exe'),
                    os.path.join(localappdata, 'Programs', 'Python', 'Python310', 'python.exe')
                ]
                for path in python_paths:
                    if os.path.exists(path):
                        try:
                            output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                            version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                            if version_match:
                                version = version_match.group(1)
                                if version.startswith('3.11') or version.startswith('3.10'):
                                    return path, version
                        except:
                            continue
                
                # Then check Microsoft Store paths (fallback)
                ms_store_paths = [
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'python.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'python3.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0', 'python.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0', 'python.exe')
                ]
                for path in ms_store_paths:
                    if os.path.exists(path):
                        try:
                            output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                            version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                            if version_match:
                                version = version_match.group(1)
                                if version.startswith('3.11') or version.startswith('3.10'):
                                    return path, version
                        except:
                            continue

            # Method 2: Try multiple username detection methods (backup approach)
            usernames = set()
            try:
                usernames.add(os.getlogin())
                usernames.add(os.environ.get('USERNAME'))
                usernames.add(os.path.expanduser('~').split('\\')[-1])
            except:
                pass

            # Add an alternative approach for paths with spaces
            user_home = os.path.expanduser('~')
            if user_home and os.path.exists(user_home):
                appdata_path = os.path.join(user_home, 'AppData', 'Local')
                if os.path.exists(appdata_path):
                    # First check Python.org installations (priority)
                    python_paths_alt = [
                        os.path.join(appdata_path, 'Programs', 'Python', 'Python311', 'python.exe'),
                        os.path.join(appdata_path, 'Programs', 'Python', 'Python310', 'python.exe')
                    ]
                    
                    for path in python_paths_alt:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue
                    
                    # Then check Microsoft Store paths (fallback)
                    ms_store_paths_alt = [
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'python.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'python3.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0', 'python.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0', 'python.exe')
                    ]
                    for path in ms_store_paths_alt:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue

            for username in usernames:
                if username:
                    # First check Python.org installations (priority)
                    python_paths = [
                        f"C:/Users/{username}/AppData/Local/Programs/Python/Python311/python.exe",
                        f"C:/Users/{username}/AppData/Local/Programs/Python/Python310/python.exe"
                    ]
                    for path in python_paths:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue
                    
                    # Then check Microsoft Store paths (fallback)
                    ms_store_paths = [
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/python.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/python3.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/python.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/python.exe"
                    ]
                    for path in ms_store_paths:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue

            # Method 3: Check standard installation locations (final fallback)
            standard_paths = [
                "C:/Python311/python.exe",
                "C:/Program Files/Python311/python.exe",
                "C:/Program Files (x86)/Python311/python.exe",
                "C:/Python310/python.exe",
                "C:/Program Files/Python310/python.exe",
                "C:/Program Files (x86)/Python310/python.exe"
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    try:
                        output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return path, version
                    except:
                        continue

            # Final attempt: Try to find python in PATH
            try:
                result = subprocess.run("where python", capture_output=True, text=True, shell=True)
                if result.returncode == 0 and result.stdout.strip():
                    python_path = result.stdout.strip().split("\n")[0]
                    if os.path.exists(python_path):
                        output = subprocess.check_output([python_path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return python_path, version
            except:
                pass
                
            # Also try python3 in PATH
            try:
                result = subprocess.run("where python3", capture_output=True, text=True, shell=True)
                if result.returncode == 0 and result.stdout.strip():
                    python_path = result.stdout.strip().split("\n")[0]
                    if os.path.exists(python_path):
                        output = subprocess.check_output([python_path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return python_path, version
            except:
                pass

        return None, None  # Return None for both path and version if no matching executable is found

    def check_venv_python_version(self, venv_path):
        if platform.system() == 'Windows':
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            python_exe = os.path.join(venv_path, 'bin', 'python')
            creation_flags = 0

        if not os.path.exists(python_exe):
            self.logger.log(f"Python executable not found in virtual environment at {venv_path}", level="ERROR")
            return

        try:
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                version_output = subprocess.check_output(
                    [python_exe, "--version"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    creationflags=creation_flags
                ).strip()
            else:
                # Fixed version for macOS - removed stdout argument
                version_output = subprocess.check_output(
                    [python_exe, "--version"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT
                ).strip()

            version_match = re.search(r'Python (\d+\.\d+\.\d+)', version_output) 
            if version_match:
                python_version = version_match.group(1)
                self.logger.log(f"Python version in venv: {python_version}", level="DEBUG")
                return python_version
            else:
                self.logger.log(f"Unable to parse Python version from output: {version_output}", level="ERROR")
                return None
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Error checking Python version: {str(e)}", level="ERROR")
            return None
        except Exception as e:
            self.logger.log(f"Unexpected error checking Python version: {str(e)}", level="ERROR")
            return None
        
    def check_venv_cuda_version(self, venv_path):
        if platform.system() == 'Darwin':  # macOS
            return None  # macOS doesn't use CUDA
            
        # Windows path handling
        try:
            site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
            if not os.path.exists(site_packages):
                return None
                
            for folder in os.listdir(site_packages):
                if folder.startswith('torch-') and '+cu' in folder:
                    cuda_version_match = re.search(r'\+cu(\d+)', folder)
                    if cuda_version_match:
                        cuda_version = f'cu{cuda_version_match.group(1)}'
                        return cuda_version        
        except (FileNotFoundError, OSError):
            return None
            
        return None


    def check_system_cuda_version(self):
        try:
            cuda_version_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, universal_newlines=True)
            version_match = re.search(r'release (\d+\.\d+)', cuda_version_output)
            if version_match:
                detected_version = version_match.group(1)
                cuda_version = f'cu{detected_version.replace(".", "")}'

                version_info = {
                    'cu118': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.0 + xFormers (default build)',
                        'notes': 'Fully supported'
                    },
                    'cu121': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.0 + xFormers (default build)',
                        'notes': 'Fully supported'
                    },
                    'cu124': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.1 + xFormers 0.0.23 (cu121 build)',
                        'notes': 'Fully supported - Using cu121 build'
                    },
                    'cu128': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs.'
                    },
                    'cu129': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs.'
                    },
                    'cu130': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs. Using cu128 wheels (highest available).'
                    }
                }

                message = f"CUDA version: {detected_version}\n\n"

                if cuda_version in version_info:
                    info = version_info[cuda_version]
                    message += f"Status: {info['status']}\n"
                    message += f"Config: {info['config']}\n"
                    message += f"Notes: {info['notes']}\n\n"

                message += "Supported Versions:\n"
                message += "CUDA 12.8 (Recommended - PyTorch 2.7.0)\n"
                message += "CUDA 12.4 (Legacy)\n"
                message += "CUDA 12.1\n"
                message += "CUDA 11.8\n\n"

                if cuda_version in ['cu128', 'cu129', 'cu130']:
                    message += "Using PyTorch 2.7.0 cu128.\n\n"
                else:
                    message += "CUDA 12.8 recommended for optimal performance.\n\n"

                message += "Continue with detected version?"

                choice = ui.messageBox('CUDA Version Check', message, buttons=['Continue', 'Cancel'])
                if choice != 0:
                    self.message_box_open = False
                    return None
                return cuda_version
            else:
                raise Exception("Unable to parse CUDA version from output.")
        except Exception as e:
            message = "CUDA not detected.\n\n"
            message += "Supported Versions:\n"
            message += "CUDA 12.8 (Recommended) - PyTorch 2.7.0\n"
            message += "CUDA 12.4 (Legacy) - PyTorch 2.4.0\n"
            message += "CUDA 12.1 - PyTorch 2.4.0\n"
            message += "CUDA 11.8 - PyTorch 2.4.0\n\n"
            message += "Install a supported CUDA version to continue.\n\n"

            choice = ui.messageBox('CUDA Not Detected', message, buttons=['Download CUDA', 'Proceed Anyway', 'Cancel'])
            if choice == 0:
                import webbrowser
                webbrowser.open('https://developer.nvidia.com/cuda-12-1-0-download-archive')
                self.message_box_open = False
                return None
            elif choice == 1:
                return 'cpu'
            else:
                self.message_box_open = False
                return None

    def Installstreamdiffusion(self):
        if self.message_box_open:
            return
        self.message_box_open = True
        self.copy_sdtd_code()
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')

        if os.path.exists(venv_path):   
            cuda_version = self.check_venv_cuda_version(venv_path)
            self.check_venv_python_version(venv_path)
            if not cuda_version and platform.system() == 'Windows':
                cuda_version = self.check_system_cuda_version()       
            choice = ui.messageBox('Virtual Environment Detected',
                                'A virtual environment already exists. Do you want to update it?',
                                buttons=['Update', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return False                    
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            python_exe, pyversion = self.find_python_exe()
            if not python_exe:
                choice = ui.messageBox('Installation Error ! No Python detected.',
                                    'Python not found. Please ensure Python 3.11.9 or Python 3.10.9 is installed and accessible.\nDo you want to continue anyway?',
                                    buttons=['Continue Anyway', 'Cancel'])
                if choice == 0:
                    python_exe = 'python3'
                else: # Cancel
                    self.message_box_open = False
                    return False
            if platform.system() == 'Windows':
                # Show CUDA version selector popup
                cuda_message = (
                    "Select PyTorch CUDA version:\n\n"
                    "CUDA 12.8 - Recommended (PyTorch 2.7.0)\n"
                    "  Supports RTX 40/30/20-series and RTX 50-series\n\n"
                    "CUDA 12.4 - Legacy compatibility\n"
                    "CUDA 12.1 - Older GPUs\n"
                    "CUDA 11.8 - Older GPUs\n\n"
                    "Note: CUDA Toolkit installation not required.\n"
                    "PyTorch includes its own CUDA runtime."
                )

                cuda_choice = ui.messageBox('Select PyTorch CUDA Version',
                                          cuda_message,
                                          buttons=['CUDA 12.8 (Recommended)', 'CUDA 12.4 (Legacy)', 'CUDA 12.1', 'CUDA 11.8', 'Cancel'])

                if cuda_choice == 0:
                    cuda_version = 'cu128'
                elif cuda_choice == 1:
                    cuda_version = 'cu124'
                elif cuda_choice == 2:
                    cuda_version = 'cu121'
                elif cuda_choice == 3:
                    cuda_version = 'cu118'
                else:  # Cancel
                    self.message_box_open = False
                    return False
            else:
                cuda_version = None
        if platform.system() == 'Windows':
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion.bat')
        else:
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion.sh')

        set_base_folder = False
        if base_folder is None or base_folder == '':
            previous_base_folder = base_folder
            base_folder = project.folder
            self.ownerComp.par.Basefolder = base_folder
            set_base_folder = True

        if not os.path.exists(venv_path):
            installation_details = f"Python version: {python_exe}\nCUDA version: {cuda_version}\nBase folder for venv: {base_folder}\nInstalling packages for: StreamDiffusionTD"
            choice = ui.messageBox('Installation Details',
                                f'{installation_details}\n\nDo you want to proceed with the installation?',
                                buttons=['Install', 'Cancel'])
            if choice != 0:
                if set_base_folder:
                    self.ownerComp.par.Basefolder = previous_base_folder
                self.message_box_open = False
                return False

        # Use StreamDiffusion-installer CLI to generate and run installation batch file
        installer_dir = os.path.join(base_folder, 'StreamDiffusion-installer')
        cli_path = os.path.join(installer_dir, 'sd_installer', 'cli.py')

        if not os.path.exists(cli_path):
            ui.messageBox('Installer Not Found',
                f'StreamDiffusion-installer CLI not found at:\n{cli_path}\n\n'
                'Please run Step 1 (Clone StreamDiffusion) first to clone the installer.',
                buttons=['OK'])
            self.message_box_open = False
            return False

        # Build CLI command to generate installation batch file
        no_cache_flag = '--no-cache' if self.ownerComp.par.Nocacheinstall else ''

        try:
            # Generate the batch file using the CLI
            self.logger.log(f'Generating installation script via CLI...', level='INFO')

            cmd = [
                python_exe, '-m', 'sd_installer',
                '--base-folder', base_folder,
                'generate-bat',
                '--cuda', cuda_version,
                '--output', bat_file_path
            ]
            if no_cache_flag:
                cmd.append(no_cache_flag)

            self.logger.log(f'CLI command: {" ".join(cmd)}', level='DEBUG')

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=installer_dir
            )
            stdout, stderr = process.communicate(timeout=30)

            if process.returncode != 0:
                self.logger.log(f'CLI generate-bat failed: {stderr}', level='ERROR')
                ui.messageBox('Batch Generation Failed',
                    f'Failed to generate installation batch file:\n{stderr}',
                    buttons=['OK'])
                self.message_box_open = False
                return False

            self.logger.log(f'Installation script generated: {bat_file_path}', level='INFO')

        except subprocess.TimeoutExpired:
            self.logger.log('CLI command timed out', level='ERROR')
            ui.messageBox('Timeout', 'CLI command timed out.', buttons=['OK'])
            self.message_box_open = False
            return False
        except Exception as e:
            self.logger.log(f'CLI execution error: {e}', level='ERROR')
            ui.messageBox('Error', f'Failed to run installer CLI:\n{e}', buttons=['OK'])
            self.message_box_open = False
            return False

        # Execute the generated batch file
        self.logger.log(f'Executing installation script: {bat_file_path}', level='INFO')
        if platform.system() == 'Windows':
            creation_flags = self.get_subprocess_creationflags()
            if creation_flags:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder, creationflags=creation_flags)
            else:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)
        else:
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=base_folder)
        self.message_box_open = False   



    def print_installation_details(self, python_exe, cuda_version, base_folder):
        print("\n====================================")
        print("Installation Details:")
        print("------------------------------------")
        print(f"Python 3.10 executable: {python_exe}")
        print(f"CUDA version: {cuda_version}")
        print(f"Base folder for venv: {base_folder}")
        print("Installing packages for: StreamDiffusionTD")
        print("====================================\n")

    def check_uv_installed(self):
        """Check if UV is installed and accessible"""
        try:
            result = subprocess.run(['uv', '--version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.log(f"UV detected: {version}", level="INFO")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def install_uv(self):
        """Install UV package manager using the official standalone installer"""
        try:
            self.logger.log("Installing UV package manager...", level="INFO")

            if platform.system() == 'Windows':
                # Use PowerShell to download and run UV installer
                install_cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    self.logger.log("UV installed successfully", level="INFO")
                    return True
                else:
                    self.logger.log(f"UV installation failed: {result.stderr}", level="ERROR")
                    return False
            else:
                # macOS/Linux install
                install_cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    self.logger.log("UV installed successfully", level="INFO")
                    return True
                else:
                    self.logger.log(f"UV installation failed: {result.stderr}", level="ERROR")
                    return False

        except Exception as e:
            self.logger.log(f"Error installing UV: {e}", level="ERROR")
            return False

    def Installstreamdiffusionuv(self):
        """
        Install StreamDiffusion using UV package manager (10-100x faster than pip).
        Creates standard venv that works with existing Start_StreamDiffusion.bat launcher.
        """
        if self.message_box_open:
            return
        self.message_box_open = True

        # Check if UV is installed
        if not self.check_uv_installed():
            choice = ui.messageBox('UV Package Manager Not Found',
                                'UV provides 10-100x faster package installation.\n\n'
                                'Would you like to install UV now?',
                                buttons=['Install UV', 'Cancel'])
            if choice != 0:
                self.message_box_open = False
                return

            if not self.install_uv():
                ui.messageBox('UV Installation Failed',
                            'Could not install UV. Please install manually from:\n'
                            'https://docs.astral.sh/uv/getting-started/installation/',
                            buttons=['OK'])
                self.message_box_open = False
                return

        self.copy_sdtd_code()
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')

        if os.path.exists(venv_path):
            cuda_version = self.check_venv_cuda_version(venv_path)
            self.check_venv_python_version(venv_path)
            if not cuda_version and platform.system() == 'Windows':
                cuda_version = self.check_system_cuda_version()
            choice = ui.messageBox('Virtual Environment Detected',
                                'A virtual environment already exists. Do you want to update it using UV?',
                                buttons=['Update', 'Cancel'])
            if choice != 0:
                self.message_box_open = False
                return False
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            python_exe, pyversion = self.find_python_exe()
            if not python_exe:
                choice = ui.messageBox('Installation Error ! No Python detected.',
                                    'Python not found. Please ensure Python 3.11.9 or Python 3.10.9 is installed.\nDo you want to continue anyway?',
                                    buttons=['Continue Anyway', 'Cancel'])
                if choice == 0:
                    python_exe = 'python3'
                else:
                    self.message_box_open = False
                    return False
            if platform.system() == 'Windows':
                # Show CUDA version selector popup (same as regular install)
                cuda_message = (
                    "Select PyTorch CUDA version:\n\n"
                    "CUDA 12.8 - Recommended (PyTorch 2.7.0)\n"
                    "  Supports RTX 40/30/20-series and RTX 50-series\n\n"
                    "CUDA 12.4 - Legacy compatibility\n"
                    "CUDA 12.1 - Older GPUs\n"
                    "CUDA 11.8 - Older GPUs\n\n"
                    "Note: CUDA Toolkit installation not required.\n"
                    "PyTorch includes its own CUDA runtime."
                )

                cuda_choice = ui.messageBox('Select PyTorch CUDA Version',
                                          cuda_message,
                                          buttons=['CUDA 12.8 (Recommended)', 'CUDA 12.4 (Legacy)', 'CUDA 12.1', 'CUDA 11.8', 'Cancel'])

                if cuda_choice == 0:
                    cuda_version = 'cu128'
                elif cuda_choice == 1:
                    cuda_version = 'cu124'
                elif cuda_choice == 2:
                    cuda_version = 'cu121'
                elif cuda_choice == 3:
                    cuda_version = 'cu118'
                else:
                    self.message_box_open = False
                    return False
            else:
                cuda_version = None

        if platform.system() == 'Windows':
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion_UV.bat')
        else:
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion_UV.sh')

        set_base_folder = False
        if base_folder is None or base_folder == '':
            previous_base_folder = base_folder
            base_folder = project.folder
            self.ownerComp.par.Basefolder = base_folder
            set_base_folder = True

        installation_details = f"Python version: {python_exe}\nCUDA version: {cuda_version}\nBase folder for venv: {base_folder}\nPackage manager: UV (10-100x faster)\nInstalling packages for: StreamDiffusionTD"
        choice = ui.messageBox('Installation Details',
                            f'{installation_details}\n\nDo you want to proceed with the installation?',
                            buttons=['Install', 'Cancel'])
        if choice != 0:
            if set_base_folder:
                self.ownerComp.par.Basefolder = previous_base_folder
            self.message_box_open = False
            return False

        if platform.system() == 'Windows':
            no_cache = ""  # UV handles caching automatically
        else:
            no_cache = ""

        def get_cuda_install_command_uv(cuda_version):
            """Returns UV commands for CUDA installation"""
            cuda_configs = {
                'cu118': {
                    'torch_cmd': f"uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118",
                    'cuda_python_cmd': f"uv pip install cuda-python==11.8.7"
                },
                'cu121': {
                    'torch_cmd': f"uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121",
                    'cuda_python_cmd': f"uv pip install cuda-python==12.9.0"
                },
                'cu124': {
                    'torch_cmd': f"uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121",
                    'cuda_python_cmd': f"uv pip install cuda-python==12.9.0"
                },
                'cu128': {
                    'torch_cmd': f"uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_cmd': f"uv pip install cuda-python==12.9.0"
                },
                'cu129': {
                    'torch_cmd': f"uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_cmd': f"uv pip install cuda-python==12.9.0"
                },
                'cu130': {
                    'torch_cmd': f"uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_cmd': f"uv pip install cuda-python==12.9.0"
                }
            }

            if cuda_version not in cuda_configs:
                return None

            config = cuda_configs[cuda_version]
            return f"{config['torch_cmd']} && {config['cuda_python_cmd']}"

        if platform.system() == 'Windows':
            torch_install_cmd = get_cuda_install_command_uv(cuda_version)
            if not torch_install_cmd:
                ui.messageBox('CUDA Version Error',
                            'Unsupported CUDA version detected.\n\n'
                            'StreamDiffusion supports CUDA 11.8, 12.1, 12.4, 12.8, 12.9, 13.0',
                            buttons=['OK'])
                self.message_box_open = False
                return
        else:
            torch_install_cmd = ""

        # Generate batch file using UV instead of pip
        batch_file_content_win = f"""
        @echo off
        echo.
        echo ========================================
        echo  StreamDiffusionTD v0.3.0 Installation
        echo  Using UV Package Manager
        echo ========================================
        echo.
        cd /d "{base_folder}"
        set "PIP_DISABLE_PIP_VERSION_CHECK=1"

        if not exist "venv" (
            echo Creating Python venv at: "{base_folder}\\venv"
            uv venv venv --python "{python_exe}"
        ) else (
            echo Virtual environment already exists at: "{base_folder}\\venv"
        )

        echo Activating virtual environment...
        call "venv\\Scripts\\activate.bat"

        if "%VIRTUAL_ENV%" == "" (
            echo Failed to activate virtual environment. Check path and venv exists.
            pause
            exit /b 1
        ) else (
            echo Virtual environment activated.
        )

        echo.
        echo.
        echo [1/7] Base System Setup
        echo Installing pip, setuptools, wheel...
        uv pip install --upgrade pip setuptools wheel

        echo Installing compatible NumPy first...
        uv pip install "numpy<2.0.0"

        rem nvidia-pyindex is no longer needed - NVIDIA packages are directly accessible
        rem Skipping nvidia-pyindex (broken with pip 25.3+ and UV)

        echo.
        echo.
        echo [2/7] CUDA Stack Installation
        echo Installing PyTorch with CUDA support...
        {torch_install_cmd}

        echo.
        echo Verifying CUDA PyTorch installation...
        python -c "import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA Available: {{torch.cuda.is_available()}}'); print(f'CUDA Version: {{torch.version.cuda if torch.cuda.is_available() else \"N/A\"}}')" || echo "WARNING: PyTorch verification failed"

        echo.
        echo.
        echo [3/7] Core Dependencies
        echo Installing diffusers, transformers, accelerate...
        uv pip install --no-deps diffusers transformers accelerate omegaconf protobuf

        echo Installing missing dependencies...
        uv pip install safetensors huggingface_hub regex requests tqdm filelock packaging pyyaml

        echo.
        echo.
        echo [4/7] StreamDiffusion Installation
        echo Installing DotSimulate StreamDiffusion fork from local clone (editable mode)...
        uv pip install --no-deps -e .[tensorrt]

        echo Installing Diffusers IPAdapter...
        uv pip install --no-deps git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99

        echo.
        echo.
        echo [5/7] Computer Vision Stack
        echo Installing opencv and image processing libraries...
        uv pip install opencv-python==4.8.1.78 Pillow scipy scikit-image

        echo Installing controlnet_aux...
        uv pip install --no-deps controlnet_aux
        uv pip install timm mediapipe

        echo.
        echo.
        echo [6/7] TouchDesigner Integration
        echo Installing TouchDesigner-specific packages...
        uv pip install python-osc pywin32 fire mss einops "peft>=0.17.0"

        echo Installing matplotlib...
        uv pip install matplotlib

        echo Installing insightface (optional)...
        uv pip install insightface || echo "WARNING: insightface installation failed - optional"

        echo.
        echo.
        echo [7/7] Final Verification
        echo Fixing version conflicts...
        uv pip install "numpy<2.0.0" --reinstall

        echo Installing optional performance packages...
        uv pip install triton || echo "INFO: triton not available - normal on Windows"

        echo.
        echo.
        echo ========================================
        echo Verification
        echo ========================================
        echo Checking PyTorch CUDA...
        python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available!'; print(f'PyTorch {{torch.__version__}} with CUDA {{torch.version.cuda}}')" || echo "CUDA verification FAILED"

        echo Checking StreamDiffusion...
        python -c "from streamdiffusion.config import load_config; print('StreamDiffusion installed successfully')" || echo "StreamDiffusion verification FAILED"

        echo.
        echo ========================================
        echo Installation Summary
        echo ========================================
        python -m pip list | findstr /I "torch diffusers streamdiffusion xformers cuda-python"

        echo.
        echo ========================================
        echo Installation Complete
        echo ========================================
        echo Check verification results above.
        echo.
        pause
        """

        print("Writing batch file for installation...")
        with open(bat_file_path, 'w', encoding='utf-8') as bat_file:
            bat_file.write(batch_file_content_win)
        print(f"Executing batch file: {bat_file_path}")

        if platform.system() == 'Windows':
            creation_flags = self.get_subprocess_creationflags()
            if creation_flags:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder, creationflags=creation_flags)
            else:
                subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)
        else:
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=base_folder)
        self.message_box_open = False

    def Installtensorrt(self):
        if self.message_box_open:
            return
        self.message_box_open = True
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')
        # Check if 'venv' directory exists
        if os.path.exists(venv_path):
            if self.ismac:
                ui.messageBox('Installing TensorRT',
                              'This installation is only supported on Windows. TensorRT installation does not work on macOS.',
                              buttons=['OK'])
                self.message_box_open = False
                return
            choice = ui.messageBox('Installing TensorRT',
                                    'Installing TensorRT...\n\n'
                                    'Note: This is for Windows only.\n'
                                    'TensorRT needs to be turned on by setting the\n'
                                    'model\'s acceleration mode to TensorRT in the\n'
                                    'Settings 1 parameter page.',
                                    buttons=['OK', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return
        self.copy_sdtd_code()
        self.install_tensorrt()
        self.message_box_open = False

    def install_tensorrt(self):
        """
        Install TensorRT using the StreamDiffusion-installer CLI.
        """
        base_folder = self.ownerComp.par.Basefolder.eval()
        installer_dir = os.path.join(base_folder, 'StreamDiffusion-installer')
        cli_path = os.path.join(installer_dir, 'sd_installer', 'cli.py')

        if not os.path.exists(cli_path):
            ui.messageBox('Installer Not Found',
                f'StreamDiffusion-installer CLI not found at:\n{cli_path}\n\n'
                'Please run Step 1 (Clone StreamDiffusion) first.',
                buttons=['OK'])
            return

        # Generate batch file that calls CLI install-tensorrt
        bat_file_path = os.path.join(base_folder, 'Install_TensorRT.bat')
        batch_file_content = f"""@echo off
echo ========================================
echo  StreamDiffusionTD TensorRT Installation
echo ========================================
echo.
cd /d "{base_folder}"

echo Attempting to activate virtual environment...
call "venv\\Scripts\\activate.bat"

if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
) else (
    echo Virtual environment activated.
)

echo.
echo Installing TensorRT via CLI...
cd /d "{installer_dir}"
python -m sd_installer install-tensorrt

echo.
echo TensorRT installation finished
pause
"""
        self.logger.log(f'Writing TensorRT batch file: {bat_file_path}', level='INFO')
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)

        self.logger.log(f'Executing TensorRT installation...', level='INFO')
        creation_flags = self.get_subprocess_creationflags()
        if creation_flags:
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder, creationflags=creation_flags)
        else:
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)

    def Updaterepo(self):
        base_folder = self.ownerComp.par.Basefolder.eval()
        if not os.path.exists(base_folder):
            self.logger.log("StreamDiffusion folder not found. Please make sure the repository is cloned.", level="ERROR")
            return
        # Path to the virtual environment's Python executable
        if platform.system() == 'Windows':
            venv_path = os.path.join(base_folder, 'venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_path):
                venv_path = os.path.join(base_folder, '.venv', 'Scripts', 'python.exe')
        elif platform.system() == 'Darwin':
            venv_path = os.path.join(base_folder, 'venv', 'bin', 'python')
            if not os.path.exists(venv_path):
                venv_path = os.path.join(base_folder, '.venv', 'bin', 'python')
        else:
            self.logger.log("Unsupported operating system.", level="ERROR")
            return
        if not os.path.exists(venv_path):
            self.logger.log("Python executable in virtual environment not found.", level="ERROR")
            return
        try:
            if platform.system() == 'Windows':
                # Create a subprocess to run the pip install command on Windows
                command = [venv_path, '-m', 'pip', 'install', '-e', base_folder]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            elif platform.system() == 'Darwin':
                # Create a subprocess to run the pip install command on macOS
                command = [venv_path, '-m', 'pip', 'install', '-e', base_folder]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            else:
                self.logger.log("Unsupported operating system.", level="ERROR")
                return
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                self.logger.log("StreamDiffusion repository updated successfully.", level="INFO")
            else:
                self.logger.log(f"Error updating StreamDiffusion repository: {stderr}", level="ERROR")
        except Exception as e:
            self.logger.log(f"Error updating StreamDiffusion repository: {str(e)}", level="ERROR")

    def copy_sdtd_code(self):
        """
        Copies the contents of Text DATs from the 'streamdiffusionTD' base component into corresponding files in the 'streamdiffusionTD' folder in the Basefolder.
        """
        base_folder = self.ownerComp.par.Basefolder.eval()
        # Ensure the 'streamdiffusionTD' folder exists
        streamdiffusionTD_folder = os.path.join(base_folder, 'streamdiffusionTD')
        if not os.path.exists(streamdiffusionTD_folder):
            os.makedirs(streamdiffusionTD_folder)
        # Paths to the Text DATs inside the 'streamdiffusionTD' base comp
        streamdiffusionTD_comp = self.ownerComp.op('streamdiffusionTD')
        text_dat_paths = {
            'td_main': 'td_main.py',
            'td_manager': 'td_manager.py',
            'td_osc_handler': 'td_osc_handler.py',
            'td_config': 'td_config.yaml',
            # 'requirements_pc': 'requirements_pc.txt',
            'requirements_mac': 'requirements_mac.txt',
            # 'pipeline_td': 'pipeline_td.py',
            # 'wrapper_td': 'wrapper_td.py',
            # 'fx_utils': 'fx_utils.py',
            # 'attention_processor': 'attention_processor.py',
            # 'ndi_utils': 'ndi_utils.py',
            'syphon_utils': 'syphon_utils.py',
            # 'daydream_client': 'daydream_client.py',
            # 'daydream_manager': 'daydream_manager.py',
            # 'webrtc_player': 'webrtc_player.html',
            'install_tensorrt': 'install_tensorrt.py'

        }
        # Copy each Text DAT's content into a file in the 'streamdiffusionTD' folder
        for dat_name, filename in text_dat_paths.items():
            file_path = os.path.join(streamdiffusionTD_folder, filename)
            
            # Special handling for td_config.yaml - generate dynamically
            if dat_name == "td_config":
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        yaml_content = self.generate_td_config_yaml()
                        file.write(yaml_content)
                    self.logger.log(f'Generated dynamic td_config.yaml at {file_path}', level='Debug')
                except Exception as e:
                    self.logger.log(f'Error generating td_config.yaml: {str(e)}', level='ERROR')
                continue
            
            # Standard Text DAT copying for other files
            text_dat = streamdiffusionTD_comp.op(dat_name)
            if text_dat:
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        text = text_dat.text    
                        if dat_name == "td_main":
                            text = text.replace("x.x.x123454321", self.get_version())
                        file.write(text)
                    # self.logger.log(f'Copied {dat_name} to {file_path}', level='Debug')
                except Exception as e:
                    self.logger.log(f'Error writing {dat_name} to {file_path}: {str(e)}', level='ERROR')
            else:
                self.logger.log(f'{dat_name} DAT not found in streamdiffusionTD', level='ERROR')
                
#         # Create the Daydream requirements file
#         daydream_req_dest = os.path.join(streamdiffusionTD_folder, 'requirements_daydream.txt')
#         daydream_requirements_content = """requests>=2.31.0
# python-osc>=1.8.0
# numpy>=1.24.0
# syphon-python>=0.1.0; sys_platform == "darwin" """
#         try:
#             with open(daydream_req_dest, 'w', encoding='utf-8') as file:
#                 file.write(daydream_requirements_content)
#             self.logger.log(f'Created requirements_daydream.txt at {daydream_req_dest}', level='Debug')
#         except Exception as e:
#             self.logger.log(f'Error creating requirements_daydream.txt: {str(e)}', level='ERROR')

    def check_install(self, force=True):
        backend = self.ownerComp.par.Backend.eval()
        
        if not force:
            clone_done = self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✓')
            install_done = self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✓')
            tensorrt_done = self.ownerComp.par.Installtensorrt.label.endswith(' ✓')
            
            if clone_done and install_done and tensorrt_done:
                return
        
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        def update_pulse_label(pulse_par, is_completed):
            current_state = pulse_par.label.endswith(' ✓')
            if force or current_state != is_completed:
                current_label = pulse_par.label
                if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
                    current_label = current_label[:-2]
                pulse_par.label = f"{current_label} {'✓' if is_completed else '✗'}"
        def check_if_installed(pulse_par):
            is_installed = pulse_par.label.endswith(' ✓')
            return is_installed
            
        # Update labels based on backend - only if they need to change
        if backend == 'Daydream':
            # For Daydream backend
            expected_labels = {
                'Clonestreamdiffusion': '1. Download StreamDiffusion',
                'Installstreamdiffusion': '2. Install [ venv + all req ]',
                'Installtensorrt': '3. Install TensorRT [ optional ]'
            }
        else:
            # For Local backend
            expected_labels = {
                'Clonestreamdiffusion': '1. Download StreamDiffusion',
                'Installstreamdiffusion': '2. Install/Update [ venv + req ]',
                'Installtensorrt': '3. Install TensorRT [ optional ]'
            }
        
        # Only update labels if they don't match expected base labels
        for par_name, expected_label in expected_labels.items():
            current_par = getattr(self.ownerComp.par, par_name)
            current_base_label = current_par.label
            # Remove status indicators to get base label
            if current_base_label.endswith(' ✓') or current_base_label.endswith(' ✗'):
                current_base_label = current_base_label[:-2]
            
            if current_base_label != expected_label:
                # Preserve any existing status indicators
                status_suffix = ''
                if current_par.label.endswith(' ✓'):
                    status_suffix = ' ✓'
                elif current_par.label.endswith(' ✗'):
                    status_suffix = ' ✗'
                if backend == 'Daydream':                
                    current_par.label = expected_label
                else:
                    current_par.label = expected_label + status_suffix
            
        if base_folder == '':
            if backend == 'Daydream':
                # Daydream backend - no local install needed, direct to Daydream page
                expected_message = 'Daydream mode | Use Login on Daydream page'
                expected_label = ''
                expected_enable = True
                expected_readonly = True
            else:
                expected_message = 'Set Basefolder or pulse 1 to download'
                expected_label = ''
                expected_enable = True
                expected_readonly = False

            if self.ownerComp.par.Installstep != expected_message:
                self.ownerComp.par.Installstep = expected_message
            if self.ownerComp.par.Installstep.label != expected_label:
                self.ownerComp.par.Installstep.label = expected_label
            if self.ownerComp.par.Installstep.enable != expected_enable:
                self.ownerComp.par.Installstep.enable = expected_enable
            if backend == 'Daydream' and self.ownerComp.par.Installstep.readOnly != expected_readonly:
                self.ownerComp.par.Installstep.readOnly = expected_readonly

            if force or not self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Clonestreamdiffusion, False)
            if force or not self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Installstreamdiffusion, False)
            if force or not self.ownerComp.par.Installtensorrt.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Installtensorrt, False)
            return
        if backend == 'Daydream':
            # Daydream backend - no local install needed
            # Only update labels if they differ (avoid UI recook)
            if self.ownerComp.par.Clonestreamdiffusion.label != '1. Download StreamDiffusion':
                self.ownerComp.par.Clonestreamdiffusion.label = '1. Download StreamDiffusion'
            if self.ownerComp.par.Installstreamdiffusion.label != '2. Install [ venv + all req ]':
                self.ownerComp.par.Installstreamdiffusion.label = '2. Install [ venv + all req ]'
            if self.ownerComp.par.Installtensorrt.label != '3. Install TensorRT [ optional ]':
                self.ownerComp.par.Installtensorrt.label = '3. Install TensorRT [ optional ]'

            expected_message = 'Daydream mode | Use Login on Daydream page'
            expected_label = ''
            expected_enable = True
            expected_readonly = True

            if self.ownerComp.par.Installstep != expected_message:
                self.ownerComp.par.Installstep = expected_message
            if self.ownerComp.par.Installstep.enable != expected_enable:
                self.ownerComp.par.Installstep.enable = expected_enable
            if self.ownerComp.par.Installstep.label != expected_label:
                self.ownerComp.par.Installstep.label = expected_label
            if self.ownerComp.par.Installstep.readOnly != expected_readonly:
                self.ownerComp.par.Installstep.readOnly = expected_readonly

        else:
            # For Local backend, use original logic
            venv_path = os.path.join(base_folder, 'venv')
            git_path = os.path.join(base_folder, '.git')
            is_cloned = os.path.exists(git_path)
            is_installed = os.path.exists(venv_path)
            
            if force or not self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✓'):
                if force or not check_if_installed(self.ownerComp.par.Clonestreamdiffusion):
                    update_pulse_label(self.ownerComp.par.Clonestreamdiffusion, is_cloned)
                    
            if force or not self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✓'):
                if force or not check_if_installed(self.ownerComp.par.Installstreamdiffusion):
                    update_pulse_label(self.ownerComp.par.Installstreamdiffusion, is_installed)

            if is_installed:
                if force or self.ownerComp.par.Installstep.enable:
                    python_version = self.check_venv_python_version(venv_path)   
                    
                    # Only check CUDA version on Windows
                    if platform.system() == 'Windows':
                        cuda_version = self.check_venv_cuda_version(venv_path)
                        version_text = f'python {python_version} | cuda {cuda_version.strip("cu")}' if cuda_version else f'python {python_version}'
                    else:
                        version_text = f'python {python_version}'
                    
                    # Only update if values have changed
                    if self.ownerComp.par.Installstep != version_text:
                        self.ownerComp.par.Installstep = version_text
                    if self.ownerComp.par.Installstep.enable != False:
                        self.ownerComp.par.Installstep.enable = False
                    if self.ownerComp.par.Installstep.label != 'Install Info:':
                        self.ownerComp.par.Installstep.label = 'Install Info:'
                    
                if force or not self.ownerComp.par.Installtensorrt.label.endswith(' ✓'):
                    if force or not check_if_installed(self.ownerComp.par.Installtensorrt):
                        tensorrt_installed = self.check_tensorrt_installation(venv_path)
                        update_pulse_label(self.ownerComp.par.Installtensorrt, tensorrt_installed)
            else:
                expected_message = 'Hit "2. Install" pulse' if is_cloned else 'Check Basefolder... it needs to be StreamDiffusion repo'
                expected_label = 'Help'
                
                if self.ownerComp.par.Installstep != expected_message:
                    self.ownerComp.par.Installstep = expected_message
                if self.ownerComp.par.Installstep.label != expected_label:
                    self.ownerComp.par.Installstep.label = expected_label
                update_pulse_label(self.ownerComp.par.Installtensorrt, False)
    #endregion INSTALLATION

    #region CONFIG AND PRESETS
    def Writeconfig(self):
        """
        Saves the current configuration to a JSON file if self.ownerComp.par.Writeconfig is True.
        The configuration is stored in the application config directory.
        """
        if not self.ownerComp.par.Writeconfig:
            if self.ownerComp.par.Writeconfig.label != 'Save OP Config':
                self.ownerComp.par.Writeconfig.label = 'Save OP Config'
            return

        base_dir = self.get_config_dir()
        if not base_dir:
            return False
            
        config_file = os.path.join(base_dir, 'streamdiffusion_operator_config.json')
        config = {}

        # Parameters to skip
        skip_params = {'clone', 'Callbackdat', 'Log', 'Log2', 'Log3', 'Writeconfig', 'Status', 'Status2', 'Status3', 'Loadconfig'}

        # Handle sequence parameters first
        sequence_params = {
            'Promptdict': ['Concept', 'Weight'],
            'Seeddict': ['Seedval', 'Seedweight'],
            'Tindexblock': ['Step'],
            'Loradictblock': ['Lorapath', 'Weight']
        }

        for seq_name, block_params in sequence_params.items():
            if hasattr(self.ownerComp.par, seq_name):
                seq_par = getattr(self.ownerComp.par, seq_name)
                blocks_data = []
                
                for block in seq_par.sequence:
                    block_data = {}
                    for param in block_params:
                        if hasattr(block.par, param):
                            par = getattr(block.par, param)
                            block_data[param] = {
                                'value': par.eval(),
                                'mode': str(par.mode),  # Store parameter mode
                                'expr': par.expr if par.mode == ParMode.EXPRESSION else None
                            }
                    blocks_data.append(block_data)
                    
                config[seq_name] = {
                    'numBlocks': seq_par.sequence.numBlocks,
                    'blocks': blocks_data
                }

        # Save regular parameters
        for page in self.ownerComp.customPages:
            for par in page.pars:
                if par.name in skip_params or isinstance(par, ParGroup) or par.name in sequence_params:
                    continue
                    
                try:
                    config[par.name] = {
                        'value': par.eval() if isinstance(par.eval(), (str, int, float, bool, type(None))) else str(par.eval()),
                        'mode': str(par.mode),
                        'expr': par.expr if par.mode == ParMode.EXPRESSION else None
                    }
                except:
                    continue

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.log(f'Operator configuration saved to {config_file}', level='INFO')
            if self.ownerComp.par.Writeconfig.label != 'Save OP Config [ on Save ]':
                self.ownerComp.par.Writeconfig.label = 'Save OP Config [ on Save ]'
            # run("me.par.Writeconfig = False",delayFrames = 50, fromOP = self.ownerComp)
            # run("me.par.Writeconfig.label = 'Save OP Config'",delayFrames = 50, fromOP = self.ownerComp)
            return True
        except Exception as e:
            self.logger.log(f'Error saving operator configuration: {str(e)}', level='ERROR')
            return False

    def Loadconfig(self):
        """
        Loads the operator configuration from the saved JSON file.
        """
        if not self.ownerComp.par.Loadconfig:
            return
        
        base_dir = self.get_config_dir()
        if not base_dir:
            return False
                
        config_file = os.path.join(base_dir, 'streamdiffusion_operator_config.json')
        
        if not os.path.exists(config_file):
            self.logger.log('No saved configuration found', level='INFO')
            return False

        skip_params = {'clone', 'Callbackdat', 'Log', 'Log2', 'Log3', 'Writeconfig', 'Status', 'Status2', 'Status3', 'Loadconfig'}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            sequence_params = {
                'Promptdict': ['Concept', 'Weight'],
                'Seeddict': ['Seedval', 'Seedweight'],
                'Tindexblock': ['Step'],
                'Loradictblock': ['Lorapath', 'Weight']
            }

            # Restore sequence parameters
            for seq_name, block_params in sequence_params.items():
                if seq_name in config and hasattr(self.ownerComp.par, seq_name):
                    seq_data = config[seq_name]
                    seq_par = getattr(self.ownerComp.par, seq_name)
                    
                    if 'numBlocks' in seq_data:
                        seq_par.sequence.numBlocks = seq_data['numBlocks']
                    
                    if 'blocks' in seq_data:
                        for i, block_data in enumerate(seq_data['blocks']):
                            if i < seq_par.sequence.numBlocks:
                                for param_name, param_info in block_data.items():
                                    if hasattr(seq_par.sequence[i].par, param_name):
                                        try:
                                            par = getattr(seq_par.sequence[i].par, param_name)
                                            if param_info['mode'] == str(ParMode.EXPRESSION) and param_info['expr']:
                                                par.expr = param_info['expr']
                                            else:
                                                par.val = param_info['value']
                                        except Exception as e:
                                            self.logger.log(f'Error setting {seq_name}[{i}].{param_name}: {str(e)}', level='WARNING')

            # Restore regular parameters
            for param_name, param_info in config.items():
                if param_name in sequence_params or param_name in skip_params:
                    continue
                    
                if hasattr(self.ownerComp.par, param_name):
                    try:
                        par = getattr(self.ownerComp.par, param_name)
                        if param_info['mode'] == str(ParMode.EXPRESSION) and param_info['expr']:
                            par.expr = param_info['expr']
                        else:
                            par.val = param_info['value']
                    except Exception as e:
                        self.logger.log(f'Could not restore parameter {param_name}: {str(e)}', level='WARNING')

            self.Basefolder(force=True)
            self.logger.log('OP Config Loaded from file', level='INFO')
            self.ownerComp.par.Writeconfig = False
            if self.ownerComp.par.Loadconfig.label != 'OP Config Loaded ✓':
                self.ownerComp.par.Loadconfig.label = 'OP Config Loaded ✓'
            run("me.par.Loadconfig = False",delayFrames = 50, fromOP = self.ownerComp)
            run("me.par.Loadconfig.label = 'Load OP Config'",delayFrames = 50, fromOP = self.ownerComp)

            return True
                
        except Exception as e:
            self.logger.log(f'Error loading operator configuration: {str(e)}', level='ERROR')
            return False

    def Resetop(self):
        """
        Resets the StreamDiffusionTD operator to default settings after user confirmation.
        """
        choice = ui.messageBox('Reset', 'How would you like to reset the StreamDiffusionTD operator?',
                            buttons=['Reset Settings', 'Full Op Reset', 'No'])

        if choice == 0 or choice == 1:
            try:
                # Clear log
                op('Logger').par.Clearlog.pulse()

                # Reset basic parameters
                if choice == 1:  # Full Op Reset
                    self.ownerComp.par.Basefolder = ''

                self.ownerComp.par.Acceleration = 'tensorrt'
                self.ownerComp.par.Uselora = False

                # Set folder parameters to expression mode with empty expressions
                folder_params = {
                    'Sdmodelsfolder': 'me.par.Basefolder + "/models/Model"',
                    'Lorafolder': 'me.par.Basefolder + "\\\\models\\\\LoRA"',
                    'Enginefolder': 'me.par.Basefolder + "/engines/td"',
                    'Hfcache': 'me.par.Basefolder + "/models"'
                }

                for param_name, expr in folder_params.items():
                    par = getattr(self.ownerComp.par, param_name)
                    par.mode = ParMode.EXPRESSION
                    par.expr = expr

                # Reset other parameters
                self.ownerComp.par.Loradictblock0lorapath = ''
                self.ownerComp.par.Customlcm = ''
                # self.ownerComp.par.Usecustomlcm = False
                self.ownerComp.par.Lastpreset = ''
                self.update_preset_dropdown(reset=True)
                self.ownerComp.par.Modelid = 'stabilityai/sdxl-turbo'
                self.ownerComp.par.Promptdict0concept = 'default fancy banana'
                self.ownerComp.par.Seeddict0seedweight = 1
                self.ownerComp.par.Tindexblock.sequence.numBlocks = 1
                self.ownerComp.par.Tindexblock0step = 1
                self.ownerComp.par.Sethfcache = False
                self.ownerComp.par.Writeconfig = False
                self.ownerComp.par.Ipadapterimage = ''
                self.ownerComp.par.Ipadapterscale = 0.0

                # Clear local backend status table
                self.reset_local_status_table()

                # Final log clear
                op('Logger').par.Clearlog.pulse()

                reset_type = "Full Op Reset" if choice == 1 else "Settings Reset"
                self.logger.log(f'StreamDiffusionTD operator {reset_type} completed', level='INFO')

            except Exception as e:
                error_msg = f'Error during reset: {str(e)}\n\nTraceback:\n{traceback.format_exc()}'
                self.logger.log(error_msg, level='ERROR')
                ui.messageBox('Reset Error', error_msg, buttons=['OK'])


    def Basefolder(self, force=False):
        self.ownerComp.par.Installstep.enable = True
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        self.sync_model_table()
        self.update_preset_dropdown()
        self.update_engine_table(silent=True)
        self.check_install(force=True)
        if self.ownerComp.par.Basefolder.eval() == '' and force == False:
            return
        if base_folder:
            self.set_streamdiffusion_install_path(base_folder)
            # self.Loadconfig()
        elif saved_path := self.get_streamdiffusion_install_path():
            if os.path.exists(saved_path):
                self.ownerComp.par.Basefolder = saved_path
                self.Loadconfig()
        # self.Getpreprocessors()
                

    def get_config_dir(self):
        """Get the DotSimulate config directory path"""
        system = platform.system()
        if system == 'Windows':
            base_dir = os.path.join(os.getenv('APPDATA'), 'dotsimulate_TD_Tools')
        elif system == 'Darwin':  # macOS
            base_dir = os.path.expanduser('~/Library/Application Support/dotsimulate_TD_Tools')
        else:
            return None
        
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def get_streamdiffusion_install_path(self):
        """Get StreamDiffusionTD base folder path from config"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return None
                
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('base_folder')
            except json.JSONDecodeError:
                self.logger.log("Error reading config file", level='ERROR')
                return None
                    
        return None

    def set_streamdiffusion_install_path(self, base_folder):
        """Save StreamDiffusionTD base folder path to config"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return False
                
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        
        config = {
            'base_folder': base_folder,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            self.logger.log(f"Error saving config: {str(e)}", level='ERROR')
            return False



    def UnlinkBasefolder(self):
        """Remove the config file"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return
        
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        if os.path.exists(config_file):
            try:
                os.remove(config_file)
                self.ownerComp.par.Basefolder = ''
                self.logger.log("StreamDiffusionTD config unlinked", level='INFO')
            except Exception as e:
                self.logger.log(f"Error removing config: {str(e)}", level='ERROR')





    def check_tensorrt_installation(self, venv_path):
        if platform.system() != 'Windows':
            expected_label = '3. Install TensorRT [ PC only ]'
            # Only update label if it's different
            current_label = self.ownerComp.par.Installtensorrt.label
            # Remove status indicators to get base label
            if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
                current_base_label = current_label[:-2]
            else:
                current_base_label = current_label
                
            if current_base_label != expected_label:
                # Preserve any existing status indicators
                status_suffix = ''
                if current_label.endswith(' ✓'):
                    status_suffix = ' ✓'
                elif current_label.endswith(' ✗'):
                    status_suffix = ' ✗'
                self.ownerComp.par.Installtensorrt.label = expected_label + status_suffix
            return False
            
        expected_label = '3. Install TensorRT [ optional ]'
        # Only update label if it's different
        current_label = self.ownerComp.par.Installtensorrt.label
        # Remove status indicators to get base label
        if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
            current_base_label = current_label[:-2]
        else:
            current_base_label = current_label
            
        if current_base_label != expected_label:
            # Preserve any existing status indicators
            status_suffix = ''
            if current_label.endswith(' ✓'):
                status_suffix = ' ✓'
            elif current_label.endswith(' ✗'):
                status_suffix = ' ✗'
            self.ownerComp.par.Installtensorrt.label = expected_label + status_suffix

        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        
        if not os.path.exists(site_packages):
            self.logger.log(f"Site-packages directory not found at {site_packages}", level="ERROR")
            return False

        # Match both old tensorrt-X.X.X and new tensorrt-cu12 or tensorrt_cu12 patterns
        tensorrt_patterns = [
            re.compile(r'tensorrt-cu\d+[-_]\d+\.\d+\.\d+.*\.dist-info'),  # tensorrt-cu12-10.14.1.48
            re.compile(r'tensorrt_cu\d+[-_]\d+\.\d+\.\d+.*\.dist-info'),  # tensorrt_cu12-10.14.1.48
            re.compile(r'tensorrt-\d+\.\d+\.\d+.*\.dist-info')            # tensorrt-10.14.1
        ]

        for item in os.listdir(site_packages):
            for pattern in tensorrt_patterns:
                if pattern.match(item):
                    # Try to extract version from any pattern
                    version_match = re.search(r'(\d+\.\d+\.\d+)', item)
                    if version_match:
                        tensorrt_version = version_match.group(1)
                        self.logger.log(f"TensorRT version {tensorrt_version} found in venv (package: {item})", level="DEBUG")
                        return True

        self.logger.log(f"TensorRT not found in {site_packages}", level="DEBUG")
        return False

    def generate_td_config_yaml(self):
        """
        Generate td_config.yaml content dynamically from TouchDesigner parameters.
        Returns YAML string that matches current parameter settings.
        """
        # Get current parameter values
        model_id = self.ownerComp.par.Modelid.eval()
        width = round(self.ownerComp.par.Width.eval() / 8) * 8
        height = round(self.ownerComp.par.Height.eval() / 8) * 8

        # Get cfg_type and guidance_scale
        cfg_type = self.ownerComp.par.Cfgtype.eval()
        guidance_scale = self.ownerComp.par.Guidancescale.eval()

        # For cfg_type "full" or "initialize", ensure guidance_scale > 1.0 in YAML
        # (pipeline.py requires this for negative prompt embeddings to be generated)
        if cfg_type in ["full", "initialize"] and guidance_scale <= 1.0:
            guidance_scale = 1.2

        # Determine model architecture type for ControlNet/IPAdapter selection
        config_type = self.get_config_type(model_id.lower())
        
        # Build YAML content as string
        yaml_content = f"""
model_id: "{model_id}"
"""

        # Add custom LCM LoRA if not "auto"
        if hasattr(self.ownerComp.par, 'Customlcm'):
            custom_lcm = self.ownerComp.par.Customlcm.eval()
            if custom_lcm and custom_lcm != "auto":
                yaml_content += f'lcm_lora_id: "{custom_lcm}"\n'

        # Add custom VAE if not "auto"
        if hasattr(self.ownerComp.par, 'Customvae'):
            custom_vae = self.ownerComp.par.Customvae.eval()
            if custom_vae and custom_vae != "auto":
                yaml_content += f'vae_id: "{custom_vae}"\n'

        # Add LoRA dict if any LoRAs are configured
        if hasattr(self.ownerComp.par, 'Loradictblock'):
            lora_dict = {}
            for block in self.ownerComp.par.Loradictblock.sequence:
                lora_path = block.par.Lorapath.eval()
                if lora_path:  # Only add if path is not empty
                    lora_weight = block.par.Weight.eval()
                    lora_dict[lora_path] = lora_weight

            if lora_dict:  # Only add to YAML if there are LoRAs
                yaml_content += "lora_dict:\n"
                for lora_path, weight in lora_dict.items():
                    yaml_content += f'  "{lora_path}": {weight}\n'

        yaml_content += f"""
# Core StreamDiffusion parameters
t_index_list: {list(block.par.Step.eval() for block in self.ownerComp.par.Tindexblock.sequence)}
width: {width}
height: {height}
device: "cuda"
dtype: "float16"

# Generation parameters (defaults, can be updated via OSC)
guidance_scale: {guidance_scale}
num_inference_steps: 50
seed: {self.ownerComp.par.Seeddict0seedval.eval()}
delta: {self.ownerComp.par.Delta.eval()}

# Prompt configuration (supports both single and blending)
prompt: "{self.ownerComp.par.Promptdict0concept.eval()}"
negative_prompt: "{self.ownerComp.par.Negprompt.eval()}"

# Optimization settings
mode: "img2img"  # Always use img2img engines (mode switching handled at runtime)
frame_buffer_size: {self.ownerComp.par.Framesize.eval()}
use_denoising_batch: {str(self.ownerComp.par.Denoisebatch.eval()).lower()}
use_tiny_vae: true
acceleration: "{self.ownerComp.par.Acceleration.eval()}"
cfg_type: "{self.ownerComp.par.Cfgtype.eval()}"
do_add_noise: {str(self.ownerComp.par.Addnoise.eval()).lower()}
warmup: {int(self.ownerComp.par.Warmup.eval())}
use_safety_checker: {str(self.ownerComp.par.Safetychecker.eval()).lower()}
skip_diffusion: {str(self.ownerComp.par.Skipdiffusion.eval()).lower()}
compile_engines_only: {str(self.ownerComp.par.Compileengines.eval()).lower()}
build_engines_if_missing: {str(self.ownerComp.par.Buildifmissing.eval()).lower()}

# Scheduler and sampler (TCD/StreamV2V support)
scheduler: "{self.ownerComp.par.Scheduler.eval()}"
sampler: "{self.ownerComp.par.Sampler.eval()}"

# StreamV2V Cached Attention (Cattenable enables, Cattmaxframes/Cattinterval tune)
use_cached_attn: {str(self.ownerComp.par.Cattenable.eval()).lower()}
cache_maxframes: {int(self.ownerComp.par.Cattmaxframes.eval())}
cache_interval: {int(self.ownerComp.par.Cattinterval.eval())}

# Image filtering (similar frame skip)
enable_similar_image_filter: {str(self.ownerComp.par.Imagefilter.eval()).lower()}
similar_image_filter_threshold: {self.ownerComp.par.Filterthresh.eval()}
similar_image_filter_max_skip_frame: {int(self.ownerComp.par.Maxskipframe.eval())}

# HuggingFace cache directory (for model downloads)
hf_cache: "{tdu.expandPath(self.ownerComp.par.Hfcache.eval()) if self.ownerComp.par.Sethfcache.eval() else ''}"

# TensorRT engine directory
engine_dir: "{self.ownerComp.par.Enginefolder.eval()}"

"""

        # Add ControlNet configuration
        if self.ownerComp.par.Usecontrolnet.eval() and hasattr(self.ownerComp.par, 'Cn'):
            yaml_content += "# ControlNet configuration\n"
            yaml_content += "use_controlnet: true\n"
            yaml_content += "controlnets:\n"

            for block in self.ownerComp.par.Cn.sequence:
                if block.par.Enable.eval():
                    model_id = block.par.Id.eval()
                    weight = block.par.Weight.eval()
                    preprocessor = block.par.Preprocessor.eval() if hasattr(block.par, 'Preprocessor') else "canny"

                    if model_id and model_id != "none":
                        # Build preprocessor params based on preprocessor type
                        preprocessor_params = {}
                        if "tensorrt" in preprocessor.lower():
                            # Use configurable engine folder + preprocessors subdirectory,
                            # then resolve an actual .engine file (not the directory path).
                            engine_base = self.ownerComp.par.Enginefolder.eval()
                            preprocessor_dir = os.path.join(engine_base, "preprocessors")
                            engine_path = None

                            if os.path.isdir(preprocessor_dir):
                                engine_files = [
                                    f for f in os.listdir(preprocessor_dir)
                                    if f.lower().endswith(".engine")
                                ]

                                if "depth" in preprocessor.lower():
                                    preferred = [
                                        f for f in engine_files
                                        if ("depth" in f.lower() or "dpt" in f.lower())
                                    ]
                                elif "pose" in preprocessor.lower() or "openpose" in preprocessor.lower():
                                    preferred = [
                                        f for f in engine_files
                                        if ("pose" in f.lower() or "yolo" in f.lower())
                                    ]
                                else:
                                    preferred = []

                                # Do not fall back to arbitrary .engine files (e.g. cnet.engine).
                                # Only use explicitly matched engines for the selected preprocessor.
                                chosen = preferred[0] if preferred else None
                                if chosen:
                                    engine_path = os.path.join(preprocessor_dir, chosen).replace("\\", "/")

                            if engine_path:
                                preprocessor_params = {"engine_path": engine_path}
                            else:
                                # Avoid writing an invalid TensorRT directory path into YAML.
                                original_preprocessor = preprocessor
                                if preprocessor.lower() == "depth_tensorrt":
                                    preprocessor = "depth"
                                elif preprocessor.lower() == "pose_tensorrt":
                                    preprocessor = "openpose"
                                if preprocessor != original_preprocessor:
                                    self.logger.log(
                                        f"generate_td_config_yaml: '{original_preprocessor}' requested but no TRT preprocessor "
                                        f"engine found in '{preprocessor_dir}'. Falling back to '{preprocessor}'.",
                                        level='WARNING'
                                    )

                        yaml_content += f'  - model_id: "{model_id}"\n'
                        yaml_content += f'    conditioning_scale: {weight}\n'
                        yaml_content += f'    preprocessor: "{preprocessor}"\n'
                        if preprocessor_params:
                            yaml_content += f'    preprocessor_params:\n'
                            for param_key, param_value in preprocessor_params.items():
                                yaml_content += f'      {param_key}: "{param_value}"\n'
                        yaml_content += f'    enabled: true\n'
        else:
            yaml_content += "# ControlNet configuration (disabled)\n"
            yaml_content += "use_controlnet: false\n"

        yaml_content += "\n"

        # Add IPAdapter configuration
        if self.ownerComp.par.Ipadapterenable.eval():
            ipadapter_scale = self.ownerComp.par.Ipadapterscale.eval()

            # Check if FaceID mode is enabled
            is_faceid = (hasattr(self.ownerComp.par, 'Ipfaceid') and
                        self.ownerComp.par.Ipfaceid.eval())

            if config_type in ["sdxl", "sdxl-turbo"]:
                if is_faceid:
                    # SDXL FaceID model
                    ipadapter_model_path = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
                    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
                else:
                    # Regular SDXL model
                    ipadapter_model_path = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
                    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
            else:
                # SD1.5 and other models
                if is_faceid:
                    # SD1.5 FaceID model (using .bin format as per h94 repo)
                    ipadapter_model_path = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"
                    image_encoder_path = "h94/IP-Adapter/models/image_encoder"
                else:
                    # Regular SD1.5 model
                    ipadapter_model_path = "h94/IP-Adapter/models/ip-adapter_sd15.bin"
                    image_encoder_path = "h94/IP-Adapter/models/image_encoder"

            yaml_content += "# IPAdapter configuration\n"
            yaml_content += "use_ipadapter: true\n"
            yaml_content += "ipadapters:\n"
            yaml_content += f'  - ipadapter_model_path: "{ipadapter_model_path}"\n'
            yaml_content += f'    image_encoder_path: "{image_encoder_path}"\n'
            yaml_content += f'    scale: {ipadapter_scale}\n'
            yaml_content += '    enabled: true\n'

            # Add FaceID-specific configuration if enabled
            if is_faceid:
                yaml_content += '    type: faceid\n'  # CRITICAL: Must be 'type: faceid', not 'is_faceid: true'
                yaml_content += '    insightface_model_name: "buffalo_l"\n'
            else:
                yaml_content += '    type: regular\n'  # Explicit type for regular IP-Adapter
        else:
            yaml_content += "# IPAdapter configuration (disabled)\n"
            yaml_content += "use_ipadapter: false\n"

        yaml_content += "\n"

        # Add FX Processors from Fx sequence (replaces old Use* toggle system)
        # Read from Fx0processor, Fx1processor, Fx2processor... sequence
        fx_processors = []
        if hasattr(self.ownerComp.par, 'Fx0processor'):
            num_blocks = self.ownerComp.par.Fx0processor.sequence.numBlocks
            for index in range(num_blocks):
                param_name = f'Fx{index}processor'
                processor_name = getattr(self.ownerComp.par, param_name).eval()
                if processor_name and processor_name.strip():
                    fx_processors.append(processor_name)

        # Build FX context once for state lookup (reuse in gather calls)
        fx_context = "|".join(fx_processors)

        # Categorize processors by pipeline stage from metadata
        image_pre_processors = []
        latent_pre_processors = []
        latent_post_processors = []
        image_post_processors = []

        table_dat = self.ownerComp.op('table_preprocessors')
        for proc_name in fx_processors:
            stage = self._get_processor_stage_from_table(proc_name, table_dat)
            if stage == 'image_pre':
                image_pre_processors.append(proc_name)
            elif stage == 'latent_pre':
                latent_pre_processors.append(proc_name)
            elif stage == 'latent_post':
                latent_post_processors.append(proc_name)
            elif stage == 'image_post':
                image_post_processors.append(proc_name)
            else:
                latent_pre_processors.append(proc_name)

        # Generate Image Preprocessing YAML
        if image_pre_processors:
            yaml_content += """# Image Preprocessing
image_preprocessing:
  enabled: true
  processors:
"""
            for order, proc_name in enumerate(image_pre_processors, start=1):
                params = self.gather_fx_parameters_for_processor(proc_name, fx_context=fx_context)
                yaml_content += f'    - type: "{proc_name}"\n'
                yaml_content += f'      order: {order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                stage = self._get_processor_stage_from_table(proc_name, table_dat)
                if proc_name in ['feedback_transform', 'feedback'] or 'feedback' in proc_name:
                    yaml_content += f'        requires_sync_processing: true\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'

        yaml_content += "\n"

        # Generate Latent Preprocessing YAML
        if latent_pre_processors:
            yaml_content += """# Latent Preprocessing
latent_preprocessing:
  enabled: true
  processors:
"""
            for order, proc_name in enumerate(latent_pre_processors, start=1):
                params = self.gather_fx_parameters_for_processor(proc_name, fx_context=fx_context)
                yaml_content += f'    - type: "{proc_name}"\n'
                yaml_content += f'      order: {order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'

        yaml_content += "\n"

        # Generate Latent Postprocessing YAML
        if latent_post_processors:
            yaml_content += """# Latent Postprocessing
latent_postprocessing:
  enabled: true
  processors:
"""
            for order, proc_name in enumerate(latent_post_processors, start=1):
                params = self.gather_fx_parameters_for_processor(proc_name, fx_context=fx_context)
                yaml_content += f'    - type: "{proc_name}"\n'
                yaml_content += f'      order: {order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'

        yaml_content += "\n"

        # Generate Image Postprocessing YAML
        if image_post_processors:
            yaml_content += """# Image Postprocessing
image_postprocessing:
  enabled: true
  processors:
"""
            for order, proc_name in enumerate(image_post_processors, start=1):
                params = self.gather_fx_parameters_for_processor(proc_name, fx_context=fx_context)
                yaml_content += f'    - type: "{proc_name}"\n'
                yaml_content += f'      order: {order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'

        yaml_content += "\n"

        # Add TouchDesigner specific settings
        osc_in_port = self.ownerComp.par.Oscinport.eval()
        osc_out_port = self.ownerComp.par.Oscoutport.eval()
        stream_name = self.ownerComp.par.Streamoutname.eval()
        
        yaml_content += """# TouchDesigner specific settings
td_settings:
  # OSC communication
"""
        yaml_content += f'  osc_receive_port: {osc_in_port}\n'
        yaml_content += f'  osc_transmit_port: {osc_out_port}\n'
        yaml_content += '  \n'
        yaml_content += '  # Memory interface\n'
        yaml_content += f'  input_mem_name: "{stream_name}"\n'
        yaml_content += f'  output_mem_name: "{stream_name}_out"\n'
        yaml_content += '  \n'
        yaml_content += '  # Debug settings\n'
        yaml_content += f'  debug_mode: {str(self.ownerComp.par.Debugmode.eval()).lower()}\n'

        return yaml_content



    def Sethfcache(self):
        try:
            start = "HF model cache: "
            new_label = None

            if self.ownerComp.par.Sethfcache.eval():
                # print(self.ownerComp.par.Hfcache.mode)
                if self.ownerComp.par.Hfcache.mode == ParMode.EXPRESSION:
                    if self.ownerComp.par.Hfcache == self.ownerComp.par.Basefolder + '/models' or self.ownerComp.par.Hfcache.expr == "me.par.Basefolder + '/models'":
                        new_label = start + "StreamDiffusion/models"
                    else:
                        new_label = start + self.ownerComp.par.Hfcache.eval()
                else:
                    new_label = start + self.get_huggingface_cache_path()
            else:
                new_label = start + self.get_huggingface_cache_path()

            # Only update if label changed (avoid UI recook)
            if new_label and self.ownerComp.par.Hfheader.label != new_label:
                self.ownerComp.par.Hfheader.label = new_label
        except Exception as e:
            self.logger.log(f"Error determining HF cache path: {str(e)}", level="WARNING")

    def get_huggingface_cache_path(self):
        user_home = os.path.expanduser("~")
        default_cache_path = os.path.join(user_home, ".cache", "huggingface", "hub")
        default_cache_path = default_cache_path.replace("\\", "/")
        if os.path.exists(default_cache_path):
            return default_cache_path
        else:
            return "Hugging Face cache path does not exist."

    def get_config_type(self, model_id):
        model_id_lower = model_id.lower()
        # Check SDXL variants first (including sdxl-turbo)
        if "xl" in model_id_lower or "sdxl" in model_id_lower:
            if "turbo" in model_id_lower:
                return "sdxl-turbo" 
            return "sdxl"
        # Non-SDXL turbo models (like SD-turbo)
        if "turbo" in model_id_lower or "sdxs" in model_id_lower:
            return "sd21"
        return "sd15"

    def normalize_model_id(self, model_id):
        """
        Normalizes a model ID to extract the file name or the last folder name from a path.
        If the model ID is a simple identifier with a single '/', it returns the last segment.
        """
        if not model_id or model_id == "":
            return None
        normalized_id = os.path.basename(model_id)
        if len(normalized_id) > 15:
            normalized_id = normalized_id[:15]
        return normalized_id
    
    def Updateengines(self):
        self.update_engine_table()

    def update_engine_table(self, silent = None):
        """
        Updates the 'my_engines' table with the list of engines in the current Enginefolder installation.
        Checks for the presence of required files and logs the results.
        """
        engines_folder = self.ownerComp.par.Enginefolder.eval()
        required_files = ['unet.engine', 'vae_encoder.engine', 'vae_decoder.engine']

        my_engines_table = self.setup_table('my_engines', headers=['Engine Configuration', 'Label', 'Status', 'Missing Files'])
        my_engines_table.clear(keepFirstRow=True)

        if not os.path.exists(engines_folder):
            if not silent:
                self.logger.log(f"No engines folder found in {engines_folder}", level='WARNING')
            return

        engine_pattern = re.compile(r'--lcm_lora-.*--tiny_vae-.*--max_batch-.*--min_batch-.*--mode-.*')
        complete_engines = []
        incomplete_engines = {}

        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                engine_path = os.path.join(root, subdir)
                if not engine_pattern.search(subdir):
                    continue
                engine_files = self.get_engine_files(engine_path)
                missing_files = [file for file in required_files if file not in engine_files]
                # Check for shared VAE files if they are missing
                if 'vae_encoder.engine' in missing_files or 'vae_decoder.engine' in missing_files:
                    vae_encoder_path, vae_decoder_path = self.find_shared_vae_files(engines_folder, engine_path)
                    if vae_encoder_path and 'vae_encoder.engine' in missing_files:
                        missing_files.remove('vae_encoder.engine')
                    if vae_decoder_path and 'vae_decoder.engine' in missing_files:
                        missing_files.remove('vae_decoder.engine')

                engine_name = os.path.relpath(engine_path, engines_folder)
                label = self.generate_engine_label(engine_name)
                if not missing_files:
                    complete_engines.append(engine_name)
                    my_engines_table.appendRow([engine_name, label, 'Complete', ''])
                else:
                    incomplete_engines[engine_name] = missing_files
                    my_engines_table.appendRow([engine_name, label, 'Incomplete', ', '.join(missing_files)])
        
        self.logger.log(f"Engine table updated with {len(complete_engines)} complete engines and {len(incomplete_engines)} incomplete engines.", level='DEBUG' if silent else 'INFO')

    def get_engine_files(self, engine_path):
        """
        Gets the list of files in the given engine path.
        """
        engine_files = []
        for file in os.listdir(engine_path):
            if os.path.isfile(os.path.join(engine_path, file)):
                engine_files.append(file)
        return engine_files

    def find_shared_vae_files(self, engines_folder, current_engine_path):
        """
        Searches for shared VAE encoder and decoder files in other engine folders.
        """
        vae_encoder_path = None
        vae_decoder_path = None
        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                engine_path = os.path.join(root, subdir)
                if engine_path == current_engine_path:
                    continue
                engine_files = self.get_engine_files(engine_path)
                if 'vae_encoder.engine' in engine_files and not vae_encoder_path:
                    vae_encoder_path = os.path.join(engine_path, 'vae_encoder.engine')
                if 'vae_decoder.engine' in engine_files and not vae_decoder_path:
                    vae_decoder_path = os.path.join(engine_path, 'vae_decoder.engine')
                if vae_encoder_path and vae_decoder_path:
                    break

        return vae_encoder_path, vae_decoder_path

    def generate_engine_label(self, engine_name):
        """
        Generates a condensed label for the engine.
        New LivePeer engines are resolution-independent, so we show:
        - Model name
        - Max batch (steps)
        - Mode (img2img/txt2img)
        - FaceID flag if present
        - Tokens if present (IPAdapter)
        """
        parts = engine_name.split('--')
        base_name = parts[0]

        # Extract max_batch
        max_batch_parts = [part.split('-')[1] for part in parts if part.startswith('max_batch')]
        max_batch = max_batch_parts[0] if max_batch_parts else '?'
        steps_label = f'{max_batch} step{"s" if int(max_batch) > 1 else ""}'

        # Extract mode
        mode_parts = [part.split('-')[1] for part in parts if part.startswith('mode')]
        mode = mode_parts[0] if mode_parts else 'img2img'

        # Check for special flags
        flags = []
        if '--fid' in engine_name or '--fid--' in engine_name:
            flags.append('FaceID')

        # Check for tokens (IPAdapter)
        tokens_parts = [part for part in parts if part.startswith('tokens')]
        if tokens_parts:
            tokens_num = tokens_parts[0].replace('tokens', '')
            flags.append(f'{tokens_num}tok')

        # Build label
        label_parts = [base_name, steps_label, mode]
        if flags:
            label_parts.append(f"({', '.join(flags)})")

        return ', '.join(label_parts)

    def Loadengine(self):
        engine = self.ownerComp.par.Engine.eval()
        if not engine:
            return
        if self.set_engine_parameters(engine):
            self.logger.log(f"Engine set to {engine}", level='INFO')

    def Log(self):
        log = self.ownerComp.par.Log.eval()
        if log == 'Server stopped...' and self.ownerComp.par.Streamactive.eval():
            self.logger.log(f"Server is streaming...", level='INFO')

    def set_engine_parameters(self, engine):
        """
        Sets the parameters based on the engine details.
        New LivePeer engines are resolution-independent, so we don't set Width/Height from the engine.
        User can set any resolution within the engine's dynamic range (typically 384-1024).
        """
        engines_folder = self.ownerComp.par.Enginefolder.eval()
        engine_path = os.path.join(engines_folder, engine)

        if not os.path.exists(engine_path):
            self.logger.log(f"Engine folder {engine_path} does not exist.", level='ERROR')
            return False

        engine_details = self.parse_engine_details(engine)
        if not engine_details:
            return False

        model_base_name = engine_details['base_name']
        self.ownerComp.par.Mymodels = model_base_name

        model_id = self.get_model_id_from_json(model_base_name)
        if model_id:
            self.ownerComp.par.Modelid = model_id
        else:
            self.logger.log(f"Model ID for base name {model_base_name} not found in working_models.json.", level='ERROR')
            return False

        # Set batch size (t_index count)
        self.ownerComp.par.Tindexblock.sequence.numBlocks = engine_details['max_batch']

        # Set IPAdapter FaceID mode if detected in engine name
        if hasattr(self.ownerComp.par, 'Ipfaceid'):
            self.ownerComp.par.Ipfaceid = engine_details.get('is_faceid', False)

        # Set acceleration to TensorRT
        self.ownerComp.par.Acceleration = 'tensorrt'

        # NOTE: Width/Height are NOT set from engine - engines are resolution-independent!
        # User keeps their current resolution settings (engine supports dynamic range)

        return True

    def parse_engine_details(self, engine):
        """
        Parses the engine details from the engine string.
        New LivePeer engines are resolution-independent - they don't encode width/height in the name.
        """
        parts = engine.split('--')
        try:
            max_batch_parts = [part.split('-')[1] for part in parts if part.startswith('max_batch')]
            if not max_batch_parts:
                self.logger.log(f"Could not find max_batch in engine name: {engine}", level='ERROR')
                return None

            details = {
                'base_name': parts[0],
                'max_batch': int(max_batch_parts[0]),
                'is_faceid': '--fid' in engine or '--fid--' in engine
            }
            return details
        except (IndexError, ValueError) as e:
            self.logger.log(f"Error parsing engine details from {engine}: {e}", level='ERROR')
            return None

    def get_model_id_from_json(self, model_base_name):
        models_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json')
        if not os.path.exists(models_file_path):
            self.logger.log(f'working_models.json not found at {models_file_path}.', level='WARNING')
            return None
        with open(models_file_path, 'r') as file:
            try:
                working_models = json.load(file)
            except json.JSONDecodeError:
                self.logger.log(f'Error reading working_models.json at {models_file_path}.', level='ERROR')
                return None
        for model in working_models:
            if isinstance(model, str):
                normalized_model = os.path.normpath(model).replace("\\", "/").lower()
                normalized_base_name = os.path.normpath(model_base_name).replace("\\", "/").lower()
                if normalized_base_name in normalized_model:
                    return model
        self.logger.log(f"Model ID for base name {model_base_name} not found in working_models.json.", level='WARNING')
        return None

    def Savepreset(self, preset_name=None):
        if preset_name is None:
            simple_name = self.generate_simple_preset_name()
            op.TDResources.PopDialog.OpenDefault(
                text=f'Enter the name for the preset or use the suggested name:\n\n{simple_name}',
                title='Name Preset',
                buttons=['OK', 'Cancel'],
                callback=self._presetNameDialogChoice,
                textEntry=True,
                escButton=1,
                enterButton=0,
                escOnClickAway=True
            )
        else:
            self._saveAndHandlePreset(preset_name)

    def _presetNameDialogChoice(self, info):
        if info['button'] == "OK":
            preset_name = info['enteredText']
        else:
            return
        if preset_name == "":
            preset_name = self.generate_simple_preset_name() 
        self._saveAndHandlePreset(preset_name)

    def _saveAndHandlePreset(self, preset_name):
        # Save the preset
        preset_name = self.save_preset(preset_name)
        self.update_preset_dropdown()
        return preset_name

    def generate_simple_preset_name(self):
        model_id = self.normalize_model_id(self.ownerComp.par.Modelid.eval())
        acceleration = self.ownerComp.par.Acceleration.eval()
        numblocks = self.ownerComp.par.Tindexblock.sequence.numBlocks
        if self.ownerComp.par.Usecustomlcm.eval():
            customlcm = self.normalize_model_id(self.ownerComp.par.Customlcm.eval())
        else:
            customlcm = False
        simple_name = f"{model_id}"
        if acceleration != 'none':
            simple_name += f"_{acceleration}"
        simple_name += f"_{numblocks}steps"
        if customlcm:
            simple_name += f"_{customlcm}"
        if len(simple_name) > 65:
            simple_name = simple_name[:65]
        return simple_name

    def save_preset(self, preset_name=None, preset_file_path=None):
        if preset_name is None:
            preset_name = "Default"
        if preset_file_path is None:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
        file_exists = os.path.exists(preset_file_path)
        os.makedirs(os.path.dirname(preset_file_path), exist_ok=True)
        presets_list = self.load_presets_list(preset_file_path)
        current_preset = {"preset_name": preset_name}
        for param_name in model_preset_params:
            if param_name == "Loradictblock":
                if self.ownerComp.par.Uselora.eval():
                    current_preset[param_name] = {block.par.Lorapath.eval(): block.par.Weight.eval() for block in self.ownerComp.par.Loradictblock.sequence}
            elif param_name == "Customlcm":
                if self.ownerComp.par.Usecustomlcm.eval():
                    current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()
            elif param_name == "Customvae":
                if self.ownerComp.par.Usecustomvae.eval():
                    current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()
            elif param_name == "Tindexblock":
                current_preset[param_name] = self.ownerComp.par.Tindexblock.sequence.numBlocks
            else:
                current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()

        if not any(preset['preset_name'] == preset_name for preset in presets_list):
            presets_list.append(current_preset)
            with open(preset_file_path, 'w') as file:
                json.dump(presets_list, file, indent=4)
            self.logger.log(f'Preset saved: {preset_name}', level='INFO')
        else:
            choice = ui.messageBox('Preset Exists',
                                f'A preset named "{preset_name}" already exists. Do you want to overwrite it?',
                                buttons=['Overwrite', 'Cancel'])
            if choice == 0:
                presets_list = [preset for preset in presets_list if preset['preset_name'] != preset_name]
                presets_list.append(current_preset)
                with open(preset_file_path, 'w') as file:
                    json.dump(presets_list, file, indent=4)
                self.logger.log(f'Preset overwritten: {preset_name}', level='INFO')
        if not file_exists:
            # Use 'model_preset_params' to generate the parameter list
            param_list = '\n'.join(model_preset_params)
            ui.messageBox('First Time Setup',
                        f'The following parameters are stored in the presets:\n{param_list}',
                        buttons=['OK'])
        return preset_name

    def load_preset(self, preset_name=None, preset_file_path=None):
        if preset_name is None:
            preset_name = "Default"
        if preset_file_path is None:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
        presets_list = self.load_presets_list(preset_file_path)
        preset = next((p for p in presets_list if p['preset_name'] == preset_name), None)
        if preset is not None:
            for param_name in model_preset_params:
                if param_name == "Loradictblock":
                    loradict = preset.get(param_name, {})
                    lora_sequence = self.ownerComp.par.Loradictblock.sequence
                    if loradict:
                        lora_sequence.numBlocks = len(loradict)
                        for i, (block_path, weight) in enumerate(loradict.items()):
                            lora_sequence[i].par.Lorapath = block_path
                            lora_sequence[i].par.Weight = weight
                elif param_name == "Tindexblock":
                    self.ownerComp.par.Tindexblock.sequence.numBlocks = preset.get(param_name, self.ownerComp.par.Tindexblock.sequence.numBlocks)
                elif param_name == "Modelid":
                    try:
                        model_name = preset.get(param_name, getattr(self.ownerComp.par, param_name).default)
                        self.ownerComp.par.Mymodels = model_name
                        # print(model_name)
                    except:
                        continue
                    setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
                elif param_name.startswith("Cn"):
                    #if Cnfolder not in preset, skip setting    
                    if param_name in preset:
                        setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
                else:
                    setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
            self.logger.log(f'Preset loaded: {preset_name}', level='INFO')
            self.ownerComp.par.Lastpreset = preset_name
        else:
            self.logger.log(f'Preset with name "{preset_name}" not found.', level='WARNING')
            self.update_preset_dropdown()

    def load_presets_list(self, preset_file_path):
        if os.path.exists(preset_file_path):
            with open(preset_file_path, 'r') as file:
                return json.load(file)
        else:
            return []

    def Loadpreset(self, preset_name=None):
        if preset_name is None:
            preset_name = self.ownerComp.par.Presetname.eval()
        self.load_preset(preset_name)

    def update_preset_dropdown(self, reset=False):
        if reset:
            # Reset the Presetname menu to a default message
            self.ownerComp.par.Presetname.menuNames = ['Saved Model Configs Will Show Here']
            self.ownerComp.par.Presetname.menuLabels = ['Saved Model Configs Will Show Here']
        else:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
            presets_list = self.load_presets_list(preset_file_path)
            preset_names = [preset['preset_name'] for preset in presets_list]
            if preset_names:
                self.ownerComp.par.Presetname.menuNames = preset_names
                self.ownerComp.par.Presetname.menuLabels = preset_names
            else:
                self.ownerComp.par.Presetname.menuNames = ['Saved Model Configs Will Show Here']
                self.ownerComp.par.Presetname.menuLabels = ['Saved Model Configs Will Show Here']

    def Savemodel(self, model=None):
        # Only save models when using local backend
        backend = self.ownerComp.par.Backend.eval()
        if backend == 'Daydream':
            self.logger.log("Skipping save_model - Daydream backend doesn't need model saving", level='DEBUG')
            return

        if self.ownerComp.par.Basefolder.eval() == '':
            return
        models_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json')
        os.makedirs(os.path.dirname(models_file_path), exist_ok=True)
        models_list = self.load_models_list(models_file_path)
        # Function to check if a string is a Windows absolute path
        def is_windows_abs_path(s):
            return len(s) > 2 and s[1] == ':' and (s[0].isalpha() and s[2] == '\\' or s[2] == '/')
        # Get the current model ID/PATH and normalize it if it's a full Windows path
        if not model:
            current_model_id = self.ownerComp.par.Modelid.eval()
            if is_windows_abs_path(current_model_id):
                current_model_id = os.path.normpath(current_model_id)
        else:
            current_model_id = model
        # Normalize all paths in the models_list for comparison, only if they are full Windows paths
        normalized_models_list = [os.path.normpath(model_id) if is_windows_abs_path(model_id) else model_id for model_id in models_list]
        # Check if the current model ID/PATH is not in the normalized list
        if current_model_id not in normalized_models_list:
            models_list.append(current_model_id)  # Append the original ID/path
            with open(models_file_path, 'w') as file:
                json.dump(models_list, file, indent=4)
            self.logger.log(f'Model ID {current_model_id} added to the working models list.', level='DEBUG')
        else:
            self.logger.log(f'Model ID {current_model_id} is already in the working models list.', level='DEBUG')
        self.sync_model_table()

    def sync_model_table(self):
        """
        Synchronizes the model IDs from the working_models.json file, models/Model folder, models/checkpoints folder,
        and the custom SD models folder with the 'model_table' DAT. Adds a second column 'Label' to the table, 
        which contains only the file name if the model path is a full file path, or the base model path/id if it's 
        a relative path or not an actual file. Ensures only unique models are added to the table.
        """
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        if os.path.exists(base_folder):
            models_file_path = os.path.join(base_folder, 'StreamDiffusionTD', 'working_models.json')
            models_list = self.load_models_list(models_file_path)

            # Ensure Model and checkpoints folders exist
            model_folder = os.path.join(base_folder, 'models', 'Model')
            checkpoints_folder = os.path.join(base_folder, 'models', 'checkpoints')
            os.makedirs(model_folder, exist_ok=True)
            os.makedirs(checkpoints_folder, exist_ok=True)

            # Get models from models/Model folder
            models_list.extend(self.get_models_from_folder(model_folder))

            # Get models from models/checkpoints folder
            models_list.extend(self.get_models_from_folder(checkpoints_folder))

            # Get models from custom SD models folder
            sd_models_folder = tdu.expandPath(self.ownerComp.par.Sdmodelsfolder.eval())
            if os.path.exists(sd_models_folder) and sd_models_folder != model_folder and sd_models_folder != checkpoints_folder:
                models_list.extend(self.get_models_from_folder(sd_models_folder))

            # Remove duplicates while preserving order
            unique_models = []
            seen = set()
            for model in models_list:
                model_key = os.path.basename(model)  # Use filename as key for uniqueness
                if model_key not in seen:
                    unique_models.append(model)
                    seen.add(model_key)

            # Get the 'model_table' DAT operator
            model_table = op('model_table')
            # Clear the table before adding new data
            model_table.clear()
            # Add headers
            model_table.appendRow(['Name', 'Label'])
            model_table.appendRow(['select_working_model_from_dropdown', 'Select Working Model from Dropdown'])

            # Add each unique model ID as a new row in the table
            for model_id in unique_models:
                # Determine the label based on whether the model_id is a full file path
                if os.path.isabs(model_id) and os.path.isfile(model_id):
                    # Extract the file name from the full file path
                    label = os.path.basename(model_id)
                else:
                    # Use the base model path/id as the label
                    label = os.path.basename(model_id)
                model_table.appendRow([model_id, label])
            
            self.logger.log('Model table synchronized with unique models from working models list and all local model folders.', level='DEBUG')
    def get_models_from_folder(self, folder_path):
        """
        Helper function to get model files from a specified folder.
        """
        model_extensions = ['.ckpt', '.safetensors', '.pt', '.pth']
        models = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if any(file.endswith(ext) for ext in model_extensions):
                    models.append(os.path.join(folder_path, file))
        return models

    def load_models_list(self, models_file_path):
        """
        Helper function that loads the list of models from the specified JSON file.
        If the file doesn't exist, it creates it with default presets.

        Parameters:
        models_file_path (str): The file path to the JSON file containing the models list.

        Returns:
        list: The list of models loaded from the file, or default presets if the file doesn't exist.
        """
        default_presets = [
            "stabilityai/sd-turbo",
            "prompthero/openjourney-v4",
            "stabilityai/sdxl-turbo"
        ]
        # Check if the file exists
        if not os.path.exists(models_file_path):
            # If the file doesn't exist, create it with default presets
            os.makedirs(os.path.dirname(models_file_path), exist_ok=True)
            with open(models_file_path, 'w') as file:
                json.dump(default_presets, file, indent=4)
            self.logger.log(f'Created working_models.json with default presets at {models_file_path}', level='INFO')
            return default_presets
        # If the file exists, read the current list of models
        with open(models_file_path, 'r') as file:
            try:
                models_list = json.load(file)
            except json.JSONDecodeError:
                # If there's a JSON decode error, use default presets
                self.logger.log(f'Error decoding JSON in {models_file_path}. Using default presets.', level='WARNING')
                models_list = default_presets
        return models_list

    def Mymodels(self):
        modelpreset = self.ownerComp.par.Mymodels.eval()
        if modelpreset != 'select_working_model_from_dropdown':
            self.ownerComp.par.Modelid = modelpreset

    def Modelid(self):
        """Called when Modelid parameter changes (TD automatic callback)"""
        model_id = self.ownerComp.par.Modelid.eval()

        # Update CN ID menus
        self.update_cn_id_menus()

        # Auto-update CN values if enabled
        if hasattr(self.ownerComp.par, 'Autoupdatecnvalues') and self.ownerComp.par.Autoupdatecnvalues:
            self.auto_update_cn_values_for_model(model_id)

        
    def Updateloramodelpath(self, block, path):
        setattr(self.ownerComp.par, f'Loradictblock{block}lorapath', path)

    def Openvenv(self):
        base_folder = self.ownerComp.par.Basefolder.eval()
        # Check if 'venv' or '.venv' directory exists
        if os.path.exists(os.path.join(base_folder, 'venv')):
            venv_path = os.path.join(base_folder, 'venv')
        elif os.path.exists(os.path.join(base_folder, '.venv')):
            venv_path = os.path.join(base_folder, '.venv')
        else:
            self.logger.log("Virtual environment not found in the specified base folder.", level="ERROR")
            return

        if platform.system() == 'Windows':
            # Windows-specific activation
            activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
            if not os.path.exists(activate_script_path):
                self.logger.log("activate.bat script not found in the virtual environment.", level="ERROR")
                return
            cmd_command = f'start cmd.exe /K "cd /d {base_folder} & {activate_script_path}"'
            try:
                subprocess.Popen(cmd_command, shell=True)
                self.logger.log("Virtual environment activated in a new CMD window.", level="INFO")
            except Exception as e:
                self.logger.log(f"Error opening CMD window and activating virtual environment: {e}", level="ERROR")
        else:
            # macOS-specific activation
            activate_script_path = os.path.join(venv_path, 'bin', 'activate')
            if not os.path.exists(activate_script_path):
                self.logger.log("activate script not found in the virtual environment.", level="ERROR")
                return
            terminal_command = f'tell application "Terminal" to do script "unset PYTHONPATH && cd {base_folder} && source {activate_script_path}"'
            try:
                subprocess.run(['osascript', '-e', terminal_command])
                self.logger.log("Virtual environment activated in a new Terminal window.", level="INFO")
            except Exception as e:
                self.logger.log(f"Error opening Terminal and activating virtual environment: {e}", level="ERROR")

    def Showbuiltin(self):  
        self.ownerComp.showCustomOnly = 1- self.ownerComp.par.Showbuiltin.eval()

    def setup_table(self, table_name, headers=None):
        # Check if the table exists, if not, create it
        table = self.ownerComp.op(table_name)
        if table is None:
            table = self.ownerComp.create(tableDAT, table_name)
            table.clear()
            if headers:
                table.appendRow(headers)
        return table

    def reset_local_status_table(self):
        """
        Clear all values in local_backend_status table (set to empty strings or defaults).
        Keeps the table structure intact - only clears the 'value' column (column 1).
        Resets CHOP-used fields: framecount, fps, server_active, stream_active, vram, timing, etc.
        """
        try:
            table = op('local_backend_status')
            if not table:
                self.logger.log("local_backend_status table not found!", level='ERROR')
                return

            # Clear all values in column 1 (value column)
            for row in range(table.numRows):
                table[row, 1] = ''

            # Set specific defaults for CHOP channels
            self._update_osc_table(table, 'server_active', 'false')
            self._update_osc_table(table, 'stream_active', 'false')
            self._update_osc_table(table, 'framecount', 0)
            self._update_osc_table(table, 'fps', 0)
            self._update_osc_table(table, 'vram_used', 0)
            self._update_osc_table(table, 'vram_total', 0)
            self._update_osc_table(table, 'start_time', 0)
            self._update_osc_table(table, 'end_time', 0)
            self._update_osc_table(table, 'last_session_uptime', 0)
            self._update_osc_table(table, 'last_error_time', 0)

            # Set connection_state to version info
            version = self.get_version()
            self._update_osc_table(table, 'connection_state', f"StreamDiffusionTD {version}")

            self.logger.log("Cleared all values in local_backend_status table", level='DEBUG')

        except Exception as e:
            self.logger.log(f"Failed to reset local status table: {e}", level='WARNING')

    def _update_osc_table(self, table, column_name, value):
        """Update stream_osc_data table - ROW-BASED: each row is [param_name, value]"""
        if not table:
            return

        # Find row with matching parameter name in column 0
        found_row = None
        for i in range(table.numRows):
            if table[i, 0].val == column_name:
                found_row = i
                break

        # If row doesn't exist, add it
        if found_row is None:
            table.appendRow([column_name, value])
        else:
            # Update existing row's value in column 1
            table[found_row, 1] = value

    def _clear_local_status_table(self, table):
        """Clear streaming fields in local_backend_status when server stops"""
        if not table:
            return

        start_time_val = table['start_time', 1].val

        if start_time_val and str(start_time_val).strip():
            try:
                # start_time is stored in SECONDS (absTime.seconds from OSC callback)
                start_time_sec = float(start_time_val)
                # pass if start_time_sec is 0 or less
                if start_time_sec > 0:
                    end_time_sec = absTime.seconds
                    final_uptime = int(end_time_sec - start_time_sec)
                    self._update_osc_table(table, 'last_session_uptime', final_uptime)
                    self._update_osc_table(table, 'end_time', end_time_sec)
                else:
                    self._update_osc_table(table, 'last_session_uptime', 0)
                    self._update_osc_table(table, 'end_time', 0)
            except Exception as e:
                self.logger.log(f"Error calculating last session uptime: {e}", level='DEBUG')

        # Clear server and streaming state fields
        self._update_osc_table(table, 'server_active', 'false')
        self._update_osc_table(table, 'stream_active', 'false')
        self._update_osc_table(table, 'framecount', 0)
        self._update_osc_table(table, 'fps', 0)

        # Clear VRAM values (can't know after server shutdown)
        self._update_osc_table(table, 'vram_used', '-')
        self._update_osc_table(table, 'vram_total', '-')

        # DON'T clear start_time - it becomes last_start_time for reference
        # connection_state, start_time, last_session_uptime, and end_time preserved to show final state

    def setup_project_info_table(self):
        """
        Sets up a table with project information.
        """
        # Define the headers for the project info table
        headers = ['Info', 'Value']
        project_info_table = self.setup_table('projectInfoTable', headers)
        project_info_table.clear(keepFirstRow = True)
        # Populate the table with project information
        project_data = [
            ('Project Folder', project.folder),
            ('Project Name', project.name),
            ('Last Save Time', project.saveTime),
            ('TDversion', str(app.build)),
            ('OS Name at Last Save', project.saveOSName),
            ('OS Version at Last Save', project.saveOSVersion),
            ('Real-Time State', str(project.realTime)),
            ('Cook Rate', str(project.cookRate)),
            ('Licenses', ', '.join([l.type for l in licenses])),
        ]

        for info, value in project_data:
            project_info_table.appendRow([info, value])


    def create_parameter(self, par_name, par_type, page='Custom', default=None, norm_min=None, norm_max=None, size=1, menu_items=None, label=None, order=None, replace=False, section=None, clamp=True):
        '''
        Creates a custom parameter on a specified page in a TouchDesigner component, supporting a wide 
        range of parameter types. This function allows for creating basic types like 'float', 'int', 
        'str', 'bool', and 'menu', as well as node reference types such as 'op', 'comp', etc. Advanced 
        options like 'label', 'order', and 'replace' enable further customization. For 'float' and 'int' 
        types, 'size' determines the number of values associated with the parameter. The 'default' 
        parameter sets the initial value, with special handling for 'menu' types where it sets the menu 
        options. If 'replace' is set to False, the function errors if the parameter already exists.

        Parameters:
        par_name (str): The name of the parameter.
        par_type (str): The type of the parameter. Supported types are:
                        'float', 'int', 'str', 'bool', 'menu', 'op', 'comp', 'object', 'panelcomp', 
                        'top', 'chop', 'sop', 'mat', 'dat', 'xy', 'xyz', 'xyzw', 'wh', 'uv', 'uvw', 
                        'rgb', 'rgba', 'file', 'folder', 'pulse', 'momentary', 'python', 'par', 'header'.
        page (str): The page name where the parameter will be added. Defaults to 'Custom'.
        default: The default value for the parameter. For 'menu', it should be a list of strings.
        norm_min, norm_max (optional): Normalized minimum and maximum values for 'float' and 'int' types.
        size (int, optional): Number of values for 'float' and 'int' types. Defaults to 1.
        menu_items (list of str, optional): List of menu items for 'menu' type parameters.
        label (str, optional): Display label of the parameter. Defaults to par_name.
        order (int, optional): Display order of the parameter.
        replace (bool, optional): Determines whether to replace an existing parameter. Defaults to True.
                                If set to False, the function errors if the parameter already exists.
        section (bool, optional): If set to True, adds a visual separator above this parameter. Useful for organizing parameters into distinct sections on the UI.

        Returns:
        The created parameter object.
        '''
        # print(f"Creating parameter: Name: {par_name}, Type: {par_type}, Page: {page}")
        # Check if the parameter already exists
        if hasattr(self.ownerComp.par, par_name):
            if not replace:
                # If the parameter exists and 'replace' is False, skip creating the parameter
                return getattr(self.ownerComp.par, par_name)
        # Check if the page exists
        custom_page = next((p for p in self.ownerComp.customPages if p.name == page), None)
        if not custom_page:
            # If the page doesn't exist, create it
            custom_page = self.ownerComp.appendCustomPage(page)

        # Mapping for parameter creation based on type
        create_method = {
            'float': lambda: custom_page.appendFloat(par_name, label=label, size=size, order=order, replace=replace),
            'int': lambda: custom_page.appendInt(par_name, label=label, size=size, order=order, replace=replace),
            'str': lambda: custom_page.appendStr(par_name, label=label, order=order, replace=replace),
            'bool': lambda: custom_page.appendToggle(par_name, label=label, order=order, replace=replace),
            'menu': lambda: custom_page.appendMenu(par_name, label=label, order=order, replace=replace),
            'op': lambda: custom_page.appendOP(par_name, label=label, order=order, replace=replace),
            'comp': lambda: custom_page.appendCOMP(par_name, label=label, order=order, replace=replace),
            'object': lambda: custom_page.appendObject(par_name, label=label, order=order, replace=replace),
            'panelcomp': lambda: custom_page.appendPanelCOMP(par_name, label=label, order=order, replace=replace),
            'top': lambda: custom_page.appendTOP(par_name, label=label, order=order, replace=replace),
            'chop': lambda: custom_page.appendCHOP(par_name, label=label, order=order, replace=replace),
            'sop': lambda: custom_page.appendSOP(par_name, label=label, order=order, replace=replace),
            'mat': lambda: custom_page.appendMAT(par_name, label=label, order=order, replace=replace),
            'dat': lambda: custom_page.appendDAT(par_name, label=label, order=order, replace=replace),
            'xy': lambda: custom_page.appendXY(par_name, label=label, order=order, replace=replace),
            'xyz': lambda: custom_page.appendXYZ(par_name, label=label, order=order, replace=replace),
            'xyzw': lambda: custom_page.appendXYZW(par_name, label=label, order=order, replace=replace),
            'wh': lambda: custom_page.appendWH(par_name, label=label, order=order, replace=replace),
            'uv': lambda: custom_page.appendUV(par_name, label=label, order=order, replace=replace),
            'uvw': lambda: custom_page.appendUVW(par_name, label=label, order=order, replace=replace),
            'rgb': lambda: custom_page.appendRGB(par_name, label=label, order=order, replace=replace),
            'rgba': lambda: custom_page.appendRGBA(par_name, label=label, order=order, replace=replace),
            'file': lambda: custom_page.appendFile(par_name, label=label, order=order, replace=replace),
            'folder': lambda: custom_page.appendFolder(par_name, label=label, order=order, replace=replace),
            'pulse': lambda: custom_page.appendPulse(par_name, label=label, order=order, replace=replace),
            'momentary': lambda: custom_page.appendMomentary(par_name, label=label, order=order, replace=replace),
            'python': lambda: custom_page.appendPython(par_name, label=label, order=order, replace=replace),
            'par': lambda: custom_page.appendPar(par_name, label=label, order=order, replace=replace),
            'header': lambda: custom_page.appendHeader(par_name, label=label, order=order, replace=replace)
        }.get(par_type.lower())

        if create_method is None:
            raise ValueError(f"Unsupported parameter type: {par_type}")

        new_param_group = create_method()

        if new_param_group is None:
            raise Exception("Parameter group creation failed")

        if not hasattr(new_param_group, 'pars'):
            raise Exception(f"Expected ParGroup, got {type(new_param_group)}")

        new_param = new_param_group[0] if new_param_group.pars else None

        if new_param is None:
            raise Exception("Parameter creation failed")
        # Set default, norm_min, norm_max, and menu_items based on the parameter type
        if default is not None:
            if par_type == 'menu' and isinstance(default, list) and all(isinstance(item, str) for item in default):
                new_param.menuNames = default
                new_param.menuLabels = default
            else:
                setattr(self.ownerComp.par, par_name, default)

        if par_type in ['float', 'int']:
            if norm_min is not None:
                new_param.normMin = norm_min
                new_param.min = norm_min
                new_param.clampMin = clamp
            if norm_max is not None:
                new_param.normMax = norm_max
                new_param.max = norm_max
                new_param.clampMax = clamp

        if section:
            new_param.startSection = True
        return new_param 
    
    def Printpardetails(self):
        """
        This function prints the details of the parameters of the owner component.
        It prints the parameter name, label, value, default value, and the page name.
        """
        print(f"{'Parameter Name':<18}|{'Label':<18}|{'Value':<18}|{'Default':<18}|{'Page':<18}")
        print("-" * 90)
        for par in self.ownerComp.customPars:
            print(f"{par.name:<18}|{par.label:<18}|{str(par.eval()):<18}|{str(par.default):<18}|{par.page.name:<18}")


    def setup_par_details_table(self):
        """
        This function sets up a table with the details of the parameters of the owner component.
        It defines the headers for the table and populates it with the parameter details.
        """
        # Define the headers for the parameter details table
        headers = ['Parameter Name', 'Label', 'Value', 'Default', 'Page', 'Norm Min', 'Norm Max', 'Min', 'Max', 'Clamp Min', 'Clamp Max', 'Enabled', 'Menu Names', 'Menu Labels']
        
        par_details_table = self.setup_table('parDetailsTable', headers)
        par_details_table.clear(keepFirstRow = True)

        # Populate the table with parameter details
        for par in self.ownerComp.customPars:
            par_details = [
                par.name,
                par.label,
                str(par.eval()),
                str(par.default),
                par.page.name,
                str(par.normMin),
                str(par.normMax),
                str(par.min),
                str(par.max),
                str(par.clampMin),
                str(par.clampMax),
                str(par.enable),
                str(par.menuNames),
                str(par.menuLabels)
            ]
            par_details_table.appendRow(par_details)


    def Editcallbacksscript(self):
        viewop = self.ownerComp.par.Callbackdat.eval()
        viewop.openViewer(unique=False, borders=True)

    def Viewinstallguide(self):
        import webbrowser
        backend = self.ownerComp.par.Backend.eval()

        if backend == 'Daydream':
            # For Daydream backend, show API Key and Docs options
            choices = ['Get API Key', 'View Docs']
            urls = ['https://app.daydream.live/dashboard/api-keys', 'https://dotsimulate.com/docs/streamdiffusiontd']
            choice = ui.messageBox('Daydream Backend',
                                 "Get your Daydream API key or view the StreamDiffusionTD documentation.",
                                 buttons=choices)
            if choice in range(len(urls)):
                webbrowser.open(urls[choice])
        else:
            # For Local backend, just open the docs
            webbrowser.open('https://dotsimulate.com/docs/streamdiffusiontd')

    def send_frame_acknowledgment(self):
        """
        Sends frame acknowledgment to StreamDiffusion server for loopback synchronization.
        This ensures the server waits for TD to process each frame before continuing.
        """
        try:
            if not self.ownerComp.par.Serveractive.eval():
                return False
            
            # Check if OSC is properly set up
            osc_out = op('oscout1')
            if not osc_out:
                self.logger.log("ERROR: oscout1 operator not found!", level='ERROR')
                return False
                
            # Send frame acknowledgment
            self.send_parameter_update('frame_ack', 1, 'command')
            # self.logger.log("✓ SENT frame acknowledgment (/frame_ack) to server", level='INFO')
            return True
            
        except Exception as e:
            self.logger.log(f"✗ ERROR sending frame acknowledgment: {str(e)}", level='ERROR')
            return False

    def update_videostreamout_top(self, url):
        """
        Updates the Destination URL of the 'videostreamout1' TOP and restarts the stream.
        """
        streamout_op = self.ownerComp.op('videostreamout1')
        if streamout_op:
            # Deactivate to ensure a clean connection to the new URL
            streamout_op.par.active = False
            # Set the new destination URL
            streamout_op.par.url = url
            # Activate the stream
            streamout_op.par.active = True
            self.logger.log(f"Updated and reactivated Video Stream Out TOP ('videostreamout1') with URL: {url}", level='INFO')
        else:
            self.logger.log("Could not find a 'videostreamout1' TOP to update.", level='WARNING')
    #endregion CONFIG AND PRESETS

    #region CONTROLNET AND IPADAPTER
    def Cnblock(self):
        """
        Sends ControlNet configurations from each block in the Cn sequence to the OSC Out DAT.
        Filters out duplicate model_ids and keeps only the highest weight for each unique model.
        Backend-aware: sends appropriate format based on par.Backend setting.

        Also detects when numBlocks changes and updates menus/preprocessors accordingly.
        """
        # Check if number of blocks changed (new block added/removed)
        current_num_blocks = len(self.ownerComp.par.Cn.sequence)
        previous_num_blocks = self.ownerComp.storage.get('cn_num_blocks', 0)

        if current_num_blocks != previous_num_blocks:
            # Number of blocks changed - update menus and preprocessors for all blocks
            self.ownerComp.storage['cn_num_blocks'] = current_num_blocks
            self.update_cn_id_menus()  # This will also auto-update preprocessors if Autopreprocess is enabled

            # Warn user if streaming to Daydream - model changes require pipeline reload
            if self.ownerComp.par.Streamactive and self.ownerComp.par.Backend.eval() == 'Daydream':
                self.logger.log(
                    f"⚠️ Must reload stream: CN blocks changed ({previous_num_blocks}→{current_num_blocks}). Only weight updates live.",
                    level='WARNING'
                )

        if not self.ownerComp.par.Streamactive:
            return

        backend = self.ownerComp.par.Backend.eval()

        if not self.ownerComp.par.Usecontrolnet.eval():
            self.send_parameter_update('use_controlnet', False)
            return

        # Unified path for all backends: emit full ControlNet list via /controlnets.
        # Keep one entry per enabled CN block (no model_id dedup).
        final_configs = []

        for block in self.ownerComp.par.Cn.sequence:
            model_id_raw = block.par.Id.eval()
            weight = block.par.Weight.eval()
            enabled = block.par.Enable.eval()

            if not enabled or not model_id_raw:
                continue

            model_id = self.resolve_controlnet_model_id(model_id_raw)

            auto_preprocess = self.ownerComp.par.Autopreprocess.eval() if hasattr(self.ownerComp.par, 'Autopreprocess') else True
            manual_preprocessor = block.par.Preprocessor.eval() if hasattr(block.par, 'Preprocessor') else None

            preprocessor = self.get_preprocessor_for_controlnet(
                model_id,
                backend,
                auto=auto_preprocess,
                manual_value=manual_preprocessor
            )

            preprocessor_params = self.gather_dynamic_cn_parameters_for_preprocessor(preprocessor)

            if not preprocessor_params:
                if "pose" in preprocessor or "openpose" in preprocessor:
                    preprocessor_params = {
                        "confidence_threshold": self.ownerComp.par.Confidencethreshold.eval() if hasattr(self.ownerComp.par, 'Confidencethreshold') else 0.5
                    }
                elif "canny" in preprocessor:
                    preprocessor_params = {
                        "low_threshold": self.ownerComp.par.Lowthreshold.eval() if hasattr(self.ownerComp.par, 'Lowthreshold') else 100,
                        "high_threshold": self.ownerComp.par.Highthreshold.eval() if hasattr(self.ownerComp.par, 'Highthreshold') else 200
                    }

            # TensorRT preprocessors require an explicit engine_path.
            # If unavailable, gracefully fall back to non-TRT preprocessors.
            if "tensorrt" in preprocessor.lower():
                has_engine_path = isinstance(preprocessor_params, dict) and bool(preprocessor_params.get("engine_path"))
                if not has_engine_path:
                    engine_base = self.ownerComp.par.Enginefolder.eval()
                    preprocessor_dir = os.path.join(engine_base, "preprocessors")
                    engine_path = None

                    if os.path.isdir(preprocessor_dir):
                        engine_files = [
                            f for f in os.listdir(preprocessor_dir)
                            if f.lower().endswith(".engine")
                        ]

                        if "depth" in preprocessor.lower():
                            preferred = [f for f in engine_files if ("depth" in f.lower() or "dpt" in f.lower())]
                        elif "pose" in preprocessor.lower() or "openpose" in preprocessor.lower():
                            preferred = [f for f in engine_files if ("pose" in f.lower() or "yolo" in f.lower())]
                        else:
                            preferred = []

                        chosen = preferred[0] if preferred else None
                        if chosen:
                            engine_path = os.path.join(preprocessor_dir, chosen).replace("\\", "/")

                    if engine_path:
                        if not isinstance(preprocessor_params, dict):
                            preprocessor_params = {}
                        preprocessor_params["engine_path"] = engine_path
                    else:
                        # No TRT preprocessor engine found; use CPU/GPU preprocessor variant.
                        original_preprocessor = preprocessor
                        if preprocessor.lower() == "depth_tensorrt":
                            preprocessor = "depth"
                        elif preprocessor.lower() == "pose_tensorrt":
                            preprocessor = "openpose"
                        if preprocessor != original_preprocessor:
                            self.logger.log(
                                f"Cnblock: '{original_preprocessor}' requested but no TRT preprocessor engine found in "
                                f"'{preprocessor_dir}'. Falling back to '{preprocessor}'.",
                                level='WARNING'
                            )

            controlnet_config = {
                "model_id": model_id,
                "conditioning_scale": float(weight),
                "enabled": True,
                "preprocessor": preprocessor,
                "preprocessor_params": preprocessor_params,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0
            }
            final_configs.append(controlnet_config)
        last_sent_enabled = self.ownerComp.storage.get('cn_use_controlnet_sent', None)

        if final_configs:
            self.logger.log(f"Cnblock: sending {len(final_configs)} controlnet(s): {json.dumps(final_configs)}", level='DEBUG')
            # Avoid redundant /use_controlnet=true pulses on every slider move.
            if last_sent_enabled is not True:
                self.send_parameter_update('use_controlnet', True)
                self.ownerComp.storage['cn_use_controlnet_sent'] = True
            self.send_parameter_update('controlnets', final_configs, 'json')
        else:
            if last_sent_enabled is not False:
                self.send_parameter_update('use_controlnet', False)
                self.ownerComp.storage['cn_use_controlnet_sent'] = False

    def Cn(self):
        """Force unified ControlNet config push on Cn sequence edits (id/weight/enable/preprocessor)."""
        self.Cnblock()

    def Ipadapterenable(self):
        """
        Sends IPAdapter enable/disable status via OSC.
        Called when Ipadapterenable parameter changes.
        """
        if not self.ownerComp.par.Streamactive:
            return
            
        enable_value = self.ownerComp.par.Ipadapterenable.eval()
        self.send_parameter_update('ipadapter_enable', enable_value)
        
        self.logger.log(f'IPAdapter enabled: {enable_value}', level='INFO')

    def Ipadapterupdate(self):
        """
        Sends IPAdapter style image update request via OSC.
        Called when Ipadapterupdate parameter is pulsed.
        """
        if not self.ownerComp.par.Streamactive:
            return

        osc_out = op('oscout1')
        self.send_parameter_update('ipadapter_update', 1, 'command')

        self.logger.log('IPAdapter style image update requested', level='INFO')

        # For Daydream backend, IP adapter is handled via parexec_daydream.py routing to submodule
        if self.ownerComp.par.Backend.eval() == 'Daydream':
            self.logger.log('Daydream IP adapter handled by submodule', level='DEBUG')

    def Ipadapterscale(self):
        """
        Updates IPAdapter scale value.
        Called when Ipadapterscale parameter changes.
        NOTE: IP adapter IS dynamic (web prototype confirms this)
        """
        if not self.ownerComp.par.Streamactive:
            return

        scale_value = self.ownerComp.par.Ipadapterscale.eval()
        self.send_parameter_update('ipadapter_scale', scale_value)

    def send_cached_attention(self):
        """
        Sends StreamV2V Cached Attention configuration.
        Called when Cattmaxframes or Cattinterval parameters change.
        These are live-updateable - backend dynamically resizes kvo_cache tensors.
        """
        if not self.ownerComp.par.Streamactive:
            return

        cache_config = {
            'cache_maxframes': int(self.ownerComp.par.Cattmaxframes.eval()),
            'cache_interval': int(self.ownerComp.par.Cattinterval.eval()),
        }
        self.send_parameter_update('cached_attention', cache_config, 'json')

    def Ipfaceid(self):
        """
        Called when Ipfaceid toggle changes.
        FaceID requires different pipeline (pip_SDXL-turbo-faceid vs pip_SDXL-turbo).
        Changing FaceID during active stream requires restart.
        """
        if self.ownerComp.par.Streamactive and self.ownerComp.par.Backend.eval() == 'Daydream':
            self.logger.log(
                "⚠️ Must reload stream: FaceID changed. Requires new pipeline.",
                level='WARNING'
            )

    def Uselatentfeedback(self):
        self.logger.log('Uselatentfeedback changed', level='INFO')
        """Called when Uselatentfeedback toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Uselatenttransform(self):
        self.logger.log('Uselatenttransform changed', level='INFO')
        """Called when Uselatenttransform toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Useprefxfeedback(self):
        self.logger.log('Useprefxfeedback changed', level='INFO')
        """Called when Useprefxfeedback toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Fxparameterupdate(self, par):
        """
        Generic callback for ALL Fx* parameters.
        Handles THREE cases:
        1. Sequence parameter change (Fx0processor, Fx1processor) → regenerate dynamic params
        2. Sequence size change (block added/removed) → regenerate dynamic params
        3. Dynamic parameter change (Fxlatentnoisenoise_strength) → send OSC
        """
        # self.logger.log(f"Fxparameterupdate: {par.name}", level='DEBUG')

        # Case 1: Sequence parameter value changed (Fx0processor, Fx1processor, etc.)
        if 'processor' in par.name.lower() and par.name.startswith('Fx') and len(par.name) > 2 and par.name[2].isdigit():
            # self.logger.log(f"Fx sequence changed: {par.name} = {par.eval()}", level='INFO')
            self.update_fx_dynamic_parameters()
            if hasattr(self.ownerComp.par, 'Fx0processor'):
                self.last_fx_sequence_size = self.ownerComp.par.Fx0processor.sequence.numBlocks
            return

        # Case 2: Detect sequence size change (block added/removed via +/- buttons)
        if hasattr(self.ownerComp.par, 'Fx0processor'):
            current_size = self.ownerComp.par.Fx0processor.sequence.numBlocks
            if hasattr(self, 'last_fx_sequence_size') and current_size != self.last_fx_sequence_size:
                self.logger.log(f"Fx sequence size changed: {self.last_fx_sequence_size} → {current_size}", level='INFO')
                self.update_fx_dynamic_parameters()
                self.last_fx_sequence_size = current_size
                return

        # Case 3: Dynamic Fx* parameter changed → save state and send OSC
        # Build FX context for state save
        fx_context_parts = []
        if hasattr(self.ownerComp.par, 'Fx0processor'):
            num_blocks = self.ownerComp.par.Fx0processor.sequence.numBlocks
            for index in range(num_blocks):
                param_name = f'Fx{index}processor'
                proc_name = getattr(self.ownerComp.par, param_name).eval()
                if proc_name and proc_name.strip():
                    fx_context_parts.append(proc_name)
        fx_context = "|".join(fx_context_parts)

        # Save parameter state (always, regardless of stream active)
        if par.name.startswith('Fx') and not ('processor' in par.name.lower() and len(par.name) > 2 and par.name[2].isdigit()):
            try:
                self.save_dyn_param_state(par.name, context=fx_context)
            except:
                pass

        # Send OSC (only when stream active)
        if not self.ownerComp.par.Streamactive:
            return

        par_name = par.name
        if not par_name.startswith('Fx'):
            # self.logger.log(f"EXIT: Not Fx param: {par_name}", level='ERROR')
            return

        clean_name = par_name[2:].lower()  # Remove 'Fx' prefix (2 chars)
        # self.logger.log(f"Processing Fx param: {par_name}, clean: {clean_name}", level='ERROR')

        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            # self.logger.log("EXIT: table_preprocessors not found", level='ERROR')
            return

        # self.logger.log(f"table_preprocessors found, rows: {table_dat.numRows}", level='ERROR')

        # Match parameter to processor
        for row in range(1, table_dat.numRows):
            processor_type = table_dat[row, 'name'].val
            params_json = table_dat[row, 'parameters_json'].val

            if not params_json or params_json == '{}':
                continue

            try:
                params_metadata = json.loads(params_json)
                for param_name in params_metadata.keys():
                    expected_name = re.sub(r'[^a-zA-Z0-9]', '', f'{processor_type}_{param_name}').lower()
                    if clean_name == expected_name:
                        osc_address = f'/fx/{processor_type}/{param_name}'
                        param_value = par.eval()
                        osc_out = self.ownerComp.op('oscout1')
                        if osc_out:
                            osc_out.sendOSC(osc_address, [param_value])
                            # self.logger.log(f"Fx OSC: {osc_address} = {param_value}", level='DEBUG')
                        return
            except Exception as e:
                self.logger.log(f"Fxparameterupdate error: {e}", level='ERROR')

    def get_preprocessor_for_controlnet(self, model_id, backend, auto=True, manual_value=None):
        """
        Get preprocessor for ControlNet model with auto-detection or manual override.

        Args:
            model_id: ControlNet model ID (e.g., "xinsir/controlnet-depth-sdxl-1.0")
            backend: 'Daydream' or 'Local'
            auto: If True, auto-detect from model_id. If False, use manual_value
            manual_value: Manual preprocessor selection (used if auto=False)

        Returns:
            preprocessor name string
        """
        if not auto and manual_value:
            return manual_value

        model_id_lower = model_id.lower()

        if "openpose" in model_id_lower or "pose" in model_id_lower:
            return "pose_tensorrt"
        elif "canny" in model_id_lower:
            return "canny"
        elif "depth" in model_id_lower:
            return "depth_tensorrt"
        elif "hed" in model_id_lower:
            return "hed"
        elif "scribble" in model_id_lower:
            return "scribble"
        elif "soft" in model_id_lower or "softedge" in model_id_lower:
            return "soft_edge"
        elif "lineart" in model_id_lower:
            return "standard_lineart"
        elif "tile" in model_id_lower:
            return "feedback"
        elif "color" in model_id_lower:
            return "passthrough"
        elif "normal" in model_id_lower:
            return "normal_bae"
        else:
            return "depth_tensorrt"

    def Refreshfx(self):
        """
        Refreshes the fxmetadata table by running async subprocess via TDAsyncIO.
        """
        asyncio_op = self.ownerComp.op('TDAsyncIO')
        if not asyncio_op:
            self.logger.log('TDAsyncIO not found - falling back to sync', level='WARNING')
            self.Getpreprocessors()
            return

        # self.logger.log('Launching async processor discovery...', level='INFO')

        # Launch async processor discovery
        asyncio_op.ext.AsyncIOManager.Run(
            self.async_get_preprocessors(),
            description="Discover FX processors",
            completion_callback=self._on_processors_discovered
        )


    def Getpreprocessors(self):
        """
        Get all available preprocessors by importing runtime registry via subprocess.
        This discovers both core processors and custom processors from custom_processors/.
        Outputs to table_preprocessors DAT with columns:
        name, is_tensorrt, display_name, description, parameters_json, use_cases
        """
        try:
            import os
            import json
            import subprocess

            base_folder = self.ownerComp.par.Basefolder.eval()
            if not base_folder:
                self.logger.log('ERROR: Basefolder parameter is not set', level='ERROR')
                return

            table_dat = self.ownerComp.op('table_preprocessors')
            if not table_dat:
                self.logger.log('ERROR: table_preprocessors DAT not found', level='ERROR')
                return

            # Subprocess to import runtime registry and extract metadata
            python_cmd = f"""
import sys
import json
import logging

# Suppress INFO logs from processor discovery
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, r'{base_folder}\\src')

try:
    from streamdiffusion.preprocessing.processors import _preprocessor_registry

    # Get list of core processors from static registry
    core_processors = set()
    init_file = r'{base_folder}\\src\\streamdiffusion\\preprocessing\\processors\\__init__.py'
    with open(init_file, 'r') as f:
        content = f.read()
        # Find static registry definition
        import re
        match = re.search(r'_preprocessor_registry\\s*=\\s*{{([^}}]+)}}', content, re.DOTALL)
        if match:
            # Extract processor names from static registry
            core_names = re.findall(r'"([^"]+)":', match.group(1))
            core_processors.update(core_names)

    # Also add conditional processors
    if 'DEPTH_TENSORRT_AVAILABLE' in content:
        core_processors.add('depth_tensorrt')
    if 'POSE_TENSORRT_AVAILABLE' in content:
        core_processors.add('pose_tensorrt')
    if 'TEMPORAL_NET_TENSORRT_AVAILABLE' in content:
        core_processors.add('temporal_net_tensorrt')
    if 'MEDIAPIPE_POSE_AVAILABLE' in content:
        core_processors.add('mediapipe_pose')
    if 'MEDIAPIPE_SEGMENTATION_AVAILABLE' in content:
        core_processors.add('mediapipe_segmentation')

    result = []
    for name, proc_class in _preprocessor_registry.items():
        try:
            metadata = proc_class.get_preprocessor_metadata()
            is_custom = name not in core_processors
            result.append({{
                'name': name,
                'display_name': metadata.get('display_name', name),
                'description': metadata.get('description', ''),
                'parameters': metadata.get('parameters', {{}}),
                'use_cases': metadata.get('use_cases', []),
                'is_custom': is_custom
            }})
        except Exception as e:
            # Skip processors that fail metadata extraction
            pass

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""

            # Run subprocess using venv Python
            venv_python = os.path.join(base_folder, 'venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_python):
                # Fallback to system python if venv not found
                venv_python = 'python'
                self.logger.log('WARNING: venv python not found, using system python', level='WARNING')

            try:
                result = subprocess.run(
                    [venv_python, '-c', python_cmd],
                    cwd=base_folder,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    self.logger.log(f'ERROR: Subprocess failed: {result.stderr}', level='ERROR')
                    return

                # Parse JSON output (find JSON array in stdout, ignoring print statements)
                stdout = result.stdout.strip()

                # Find the JSON array (starts with [ and ends with ])
                json_start = stdout.find('[')
                if json_start == -1:
                    self.logger.log(f'ERROR: No JSON array found in output', level='ERROR')
                    self.logger.log(f'Output was: {stdout[:200]}', level='DEBUG')
                    return

                json_str = stdout[json_start:]
                processors_data = json.loads(json_str)

                if 'error' in processors_data:
                    self.logger.log(f'ERROR: Failed to load processors: {processors_data["error"]}', level='ERROR')
                    return

            except subprocess.TimeoutExpired:
                self.logger.log('ERROR: Subprocess timeout while loading processors', level='ERROR')
                return
            except json.JSONDecodeError as e:
                self.logger.log(f'ERROR: Failed to parse subprocess output: {e}', level='ERROR')
                self.logger.log(f'Output was: {result.stdout}', level='DEBUG')
                return

            # Clear and populate table
            table_dat.clear()
            table_dat.appendRow(['name', 'is_tensorrt', 'is_custom', 'display_name', 'description', 'parameters_json', 'use_cases'])

            for proc_data in sorted(processors_data, key=lambda x: x['name']):
                name = proc_data['name']
                is_trt = 'tensorrt' in name.lower() or 'trt' in name.lower()
                is_custom = proc_data.get('is_custom', False)
                display_name = proc_data.get('display_name', name)
                description = proc_data.get('description', '')

                # Convert parameters dict to JSON string
                params_dict = proc_data.get('parameters', {})
                parameters_json = json.dumps(params_dict, indent=2) if params_dict else '{}'

                # Convert use_cases list to comma-separated string
                use_cases_list = proc_data.get('use_cases', [])
                use_cases = ', '.join(use_cases_list) if use_cases_list else ''

                table_dat.appendRow([
                    name,
                    '1' if is_trt else '0',
                    '1' if is_custom else '0',
                    display_name,
                    description,
                    parameters_json,
                    use_cases
                ])

            self.logger.log(f'Loaded {len(processors_data)} preprocessors with metadata into table_preprocessors', level='INFO')

        except Exception as e:
            self.logger.log(f'ERROR getting preprocessors: {e}', level='ERROR')
            import traceback
            traceback.print_exc()

    def _extract_metadata_from_file(self, filepath):
        """
        Extract get_preprocessor_metadata() return value from a Python file
        without importing it (avoids dependency issues).
        """
        import ast

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Parse the file as AST
            tree = ast.parse(file_content)

            # Find the class definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for get_preprocessor_metadata method
                    for item in node.body:
                        if (isinstance(item, ast.FunctionDef) and
                            item.name == 'get_preprocessor_metadata'):
                            # Find the return statement
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Return) and stmt.value:
                                    # Extract the return value (should be a dict)
                                    metadata_str = ast.get_source_segment(file_content, stmt.value)
                                    if metadata_str:
                                        # Safely evaluate the dictionary literal
                                        try:
                                            metadata = ast.literal_eval(metadata_str)
                                            return metadata
                                        except:
                                            # If literal_eval fails, try regex extraction
                                            return self._extract_metadata_with_regex(file_content)

            return None

        except Exception as e:
            self.logger.log(f'WARNING: Failed to parse {filepath}: {e}', level='WARNING')
            return None

    def _extract_metadata_with_regex(self, file_content):
        """
        Fallback method to extract metadata using regex pattern matching.
        """
        try:
            # Find get_preprocessor_metadata method
            pattern = r'def get_preprocessor_metadata\(.*?\):\s*return\s*(\{.*?\n\s*\})'
            match = re.search(pattern, file_content, re.DOTALL)

            if not match:
                return None

            metadata_str = match.group(1)

            # Try to convert Python dict syntax to JSON
            # Replace single quotes with double quotes
            metadata_str = re.sub(r"'([^']*)'", r'"\1"', metadata_str)
            # Handle Python True/False/None
            metadata_str = metadata_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')

            # Parse as JSON
            metadata = json.loads(metadata_str)
            return metadata

        except Exception as e:
            self.logger.log(f'WARNING: Regex metadata extraction failed: {e}', level='WARNING')
            return None
    #endregion CONTROLNET AND IPADAPTER

    #region DYNAMIC PARAMETER SYSTEM - Dyn* parameters for preprocessors

    def update_cn_dynamic_parameters(self):
        """
        Updates dynamic Dyn* parameters for all enabled ControlNet blocks.
        Called when Autoupdatecnpars is True and CN selection changes.
        Follows dynamic parameter pattern.
        """
        if not hasattr(self.ownerComp.par, 'Autoupdatecnpars') or not self.ownerComp.par.Autoupdatecnpars.eval():
            return

        # Collect all unique preprocessors from ALL CN blocks (enabled or disabled)
        # The Preprocessor parameter is the source of truth
        active_preprocessors = set()

        # Filter out latent-domain processors (they belong on Fx page, not ControlNet page)
        latent_only_processors = ['latent_feedback', 'latent_transform']

        for block in self.ownerComp.par.Cn.sequence:
            # Get preprocessor value directly from the parameter (source of truth)
            if hasattr(block.par, 'Preprocessor'):
                preprocessor_name = block.par.Preprocessor.eval()
                if preprocessor_name and preprocessor_name.strip():
                    # Skip latent-domain processors - they're for Fx page only
                    if preprocessor_name not in latent_only_processors:
                        active_preprocessors.add(preprocessor_name)

        # Remove old Dyn* parameters first
        self._remove_all_dynamic_cn_parameters()

        # Create new Dyn* parameters for all active preprocessors
        is_very_first = True
        for preprocessor_name in active_preprocessors:
            self._create_dynamic_parameters_for_preprocessor(preprocessor_name, is_first_preprocessor=is_very_first)
            is_very_first = False

    def _remove_all_dynamic_cn_parameters(self):
        """
        Remove all Dyn* parameters (ControlNet preprocessor dynamic parameters).
        IMPORTANT: Saves state of all Dyn* parameters before removal!
        """
        params_to_remove = []

        # Find all Dyn* parameters
        for par_tuple in self.ownerComp.customPars:
            # customPars returns tuples of parameters
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            if par.name.startswith('Dyn'):
                params_to_remove.append(par)

        # Save state of all parameters BEFORE removing them
        for par in params_to_remove:
            self.save_dyn_param_state(par.name)

        # Remove them
        for par in params_to_remove:
            try:
                par.destroy()
            except Exception as e:
                self.logger.log(f'WARNING: Failed to remove parameter {par.name}: {e}', level='WARNING')

    def _create_dynamic_parameters_for_preprocessor(self, preprocessor_name, is_first_preprocessor=False):
        """
        Create Dyn* parameters for a specific preprocessor based on its metadata.
        Places parameters AFTER the Preprocessorcontrols header on ControlNet page.
        """
        # Get metadata from table_preprocessors
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return

        # Find the preprocessor row
        preprocessor_row = None
        for row in range(1, table_dat.numRows):  # Skip header
            if table_dat[row, 'name'].val == preprocessor_name:
                preprocessor_row = row
                break

        if preprocessor_row is None:
            self.logger.log(f'WARNING: Preprocessor {preprocessor_name} not found in table_preprocessors', level='WARNING')
            return

        # Get parameters_json
        parameters_json = table_dat[preprocessor_row, 'parameters_json'].val
        if not parameters_json or parameters_json == '{}':
            # No parameters for this preprocessor
            return

        try:
            params_dict = json.loads(parameters_json)
        except Exception as e:
            self.logger.log(f'ERROR: Failed to parse parameters_json for {preprocessor_name}: {e}', level='ERROR')
            return

        # Calculate base order to ensure dynamic params appear AFTER Preprocessorcontrols header
        cn_page = next((p for p in self.ownerComp.customPages if p.name == 'ControlNet'), None)
        if cn_page:
            # Find the Preprocessorcontrols header parameter order
            preprocessor_header_order = None
            for par_tuple in cn_page.pars:
                par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
                # Look for the preprocessor controls header parameter (various possible names)
                if par.name in ['Preprocessorcontrols', 'Preprocessor', 'Preprocessorheader', 'Cnpreprocessor']:
                    if hasattr(par, 'order') and par.order is not None:
                        preprocessor_header_order = par.order
                        break

            if preprocessor_header_order is not None:
                # Start dynamic params right after the header
                base_order = preprocessor_header_order + 100
            else:
                # Fallback: find max order and add offset
                existing_orders = []
                for par_tuple in cn_page.pars:
                    par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
                    if hasattr(par, 'order') and par.order is not None:
                        existing_orders.append(par.order)
                base_order = max(existing_orders) + 100 if existing_orders else 10000
        else:
            base_order = 10000

        current_order = base_order

        # Create Dyn* parameters for each parameter in the metadata
        # Only add section separator if this is NOT the very first parameter overall
        is_first_param = True
        for param_name, param_metadata in params_dict.items():
            add_section = is_first_param and not is_first_preprocessor
            self._create_single_dynamic_parameter(preprocessor_name, param_name, param_metadata, section=add_section, order=current_order)
            is_first_param = False
            current_order += 1  # Increment order for each param

    def _create_single_dynamic_parameter(self, preprocessor_name, param_name, param_metadata, section=False, page='ControlNet', prefix='Dyn', order=None, context=None):
        """
        Create a single Dyn* parameter using the existing create_parameter() method.
        Format: Dyn{preprocessor}_{paramname} (e.g., Dyncannylowthreshold)

        Args:
            context: Optional FX context string (e.g., "latent_noise|feedback_transform") for state restore
        """
        param_type = param_metadata.get('type', 'float')
        default = param_metadata.get('default', 0)
        param_range = param_metadata.get('range', None)
        description = param_metadata.get('description', '')
        options = param_metadata.get('options', None)

        # Handle string type: if it has options, use menu; otherwise use str
        if param_type == 'string':
            if options and len(options) > 0:
                param_type = 'menu'
                default = options  # For menu, default is the list of options
            else:
                param_type = 'str'

        # Create normalized parameter name (TD convention: Firstlowercase)
        # TouchDesigner STRICT rule: First letter uppercase, REST ALL LOWERCASE (no mixed case!)
        # Format: dyn_canny_low_threshold -> Dyncannylowthreshold
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
        # First letter uppercase, rest all lowercase
        normalized_name = prefix + clean_name.lower()

        # Use specified page
        page_name = page

        # Get min/max from range if available
        norm_min = param_range[0] if param_range and len(param_range) > 0 else None
        norm_max = param_range[1] if param_range and len(param_range) > 1 else None

        # For Fx transform parameters, disable clamping so users can type values outside slider range
        # Transform params: zoom, pan_x, pan_y, rotation (not noise_strength, noise_seed_mix, feedback_blend)
        transform_params = ['zoom', 'pan_x', 'pan_y', 'rotation', 'noise_strength']
        should_clamp = param_name not in transform_params

        # Create parameter using the existing create_parameter method
        try:
            new_par = self.create_parameter(
                par_name=normalized_name,
                par_type=param_type,
                page=page_name,
                default=default,
                norm_min=norm_min,
                norm_max=norm_max,
                label=param_name,
                replace=True,
                section=section,
                clamp=should_clamp,
                order=order
            )

            # Set help text if available
            if description and new_par:
                new_par.help = description

            # Try to restore saved state if it exists (with context for FX params)
            saved_state = self.restore_dyn_param_state(normalized_name, context=context)
            if saved_state:
                self.apply_dyn_param_state(normalized_name, saved_state)
                # self.logger.log(f'Restored saved state for {normalized_name}', level='INFO')

        except Exception as e:
            self.logger.log(f'ERROR: Failed to create parameter {normalized_name}: {e}', level='ERROR')

    def gather_dynamic_cn_parameters_for_preprocessor(self, preprocessor_name):
        """
        Gather current values of Dyn* parameters for a specific preprocessor.
        Returns dict of {original_param_name: current_value}
        """
        params_dict = {}

        # Get metadata to map parameter names correctly
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return params_dict

        # Find preprocessor metadata
        preprocessor_row = None
        for row in range(1, table_dat.numRows):
            if table_dat[row, 'name'].val == preprocessor_name:
                preprocessor_row = row
                break

        if not preprocessor_row:
            return params_dict

        # Get parameter names from metadata
        parameters_json = table_dat[preprocessor_row, 'parameters_json'].val
        if not parameters_json or parameters_json == '{}':
            return params_dict

        try:
            params_metadata = json.loads(parameters_json)
        except:
            return params_dict

        # For each parameter in metadata, find its Dyn* parameter and get value
        for param_name in params_metadata.keys():
            # Reconstruct the normalized name the same way it was created
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
            normalized_name = 'Dyn' + clean_name.lower()

            # Get the parameter value
            if hasattr(self.ownerComp.par, normalized_name):
                par = getattr(self.ownerComp.par, normalized_name)
                params_dict[param_name] = par.eval()

        return params_dict

    #endregion DYNAMIC PARAMETER SYSTEM

    #region Fx DYNAMIC PARAMETERS - Latent-domain processors on Fx page

    def gather_fx_parameters_for_processor(self, preprocessor_name, fx_context=None):
        """
        Gather current values of Fx* parameters for a specific Fx processor.
        Returns dict of {original_param_name: current_value}

        Args:
            preprocessor_name: Name of processor to gather params for
            fx_context: Optional pre-built context string (processor1|processor2|...) to avoid rebuilding
        """
        params_dict = {}

        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return params_dict

        # Find preprocessor metadata
        for row in range(1, table_dat.numRows):
            if table_dat[row, 'name'].val == preprocessor_name:
                parameters_json = table_dat[row, 'parameters_json'].val
                if not parameters_json or parameters_json == '{}':
                    return params_dict

                try:
                    params_metadata = json.loads(parameters_json)
                except:
                    return params_dict

                # For each parameter in metadata, find its Fx* parameter and get value
                for param_name in params_metadata.keys():
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
                    normalized_name = 'Fx' + clean_name.lower()

                    # Try to get live parameter value
                    if hasattr(self.ownerComp.par, normalized_name):
                        par = getattr(self.ownerComp.par, normalized_name)
                        try:
                            params_dict[param_name] = par.eval()
                        except:
                            # If eval fails (broken expression), try .val
                            try:
                                params_dict[param_name] = par.val
                            except:
                                # Fall back to default
                                params_dict[param_name] = par.default
                    else:
                        # Parameter doesn't exist - try cached state if context provided
                        if fx_context:
                            cached_state = self.restore_dyn_param_state(normalized_name, context=fx_context)
                            if cached_state:
                                try:
                                    params_dict[param_name] = float(cached_state['val']) if params_metadata[param_name].get('type') == 'float' else cached_state['val']
                                except:
                                    params_dict[param_name] = cached_state['val']
                            else:
                                # No cached state - use default from metadata
                                params_dict[param_name] = params_metadata[param_name].get('default', 0)
                        else:
                            # No context provided - use default
                            params_dict[param_name] = params_metadata[param_name].get('default', 0)

                return params_dict

        return params_dict

    def update_fx_dynamic_parameters(self):
        if not self.ownerComp.par.Updatefxpars.eval():
            return
        """
        Updates Fx* dynamic parameters based on FX sequence selection.
        Reads from Fx0processor, Fx1processor, Fx2processor... sequence parameters.
        """
        # Build FX context from current sequence
        active_fx = []
        if hasattr(self.ownerComp.par, 'Fx0processor'):
            num_blocks = self.ownerComp.par.Fx0processor.sequence.numBlocks
            for index in range(num_blocks):
                param_name = f'Fx{index}processor'
                processor_name = getattr(self.ownerComp.par, param_name).eval()
                if processor_name and processor_name.strip():
                    active_fx.append(processor_name)

        # Build context key for state save/restore
        fx_context = "|".join(active_fx)
        # self.logger.log(f"update_fx_dynamic_parameters: active_fx = {active_fx}, context = {fx_context}", level='DEBUG')

        # Remove old Fx* params (but preserve sequence params and menu params)
        for par_tuple in self.ownerComp.customPars:
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            # Only destroy Fx* params that are NOT:
            # - sequence params (Fx0processor, Fx1processor) - must have digit after Fx
            # - menu params (Fxprocessorsmenu0processor, etc.)
            # - static params (Updatefxpars, Refreshfx, Processor, Fx header)
            is_sequence_param = (par.name.startswith('Fx') and
                                len(par.name) > 2 and
                                par.name[2].isdigit() and
                                'processor' in par.name.lower())
            is_static_param = par.name in ['Updatefxpars', 'Refreshfx', 'Processor', 'Fx']

            if par.name.startswith('Fx') and not is_sequence_param and not is_static_param:
                self.save_dyn_param_state(par.name, context=fx_context)
                try:
                    par.destroy()
                except Exception as e:
                    self.logger.log(f"ERROR destroying Fx parameter {par.name}: {e}", level='ERROR')

        # Create Fx* params on Fx page with Fx prefix
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return

        # Calculate base order to ensure dynamic Fx params appear AFTER sequence params
        fx_page = next((p for p in self.ownerComp.customPages if p.name == 'Fx'), None)
        if fx_page:
            existing_orders = []
            for par_tuple in fx_page.pars:
                par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
                # Get order from ALL existing params on Fx page (including sequence params)
                if hasattr(par, 'order') and par.order is not None:
                    existing_orders.append(par.order)

            # Start dynamic params AFTER all existing params (including Fx0processor, Refreshfx, etc.)
            base_order = max(existing_orders) + 100 if existing_orders else 10000
        else:
            base_order = 10000

        # self.logger.log(f"Dynamic Fx params starting at order: {base_order}", level='DEBUG')

        is_first_processor = True
        current_order = base_order

        for proc_name in active_fx:
            for row in range(1, table_dat.numRows):
                if table_dat[row, 'name'].val == proc_name:
                    import json
                    params_json = table_dat[row, 'parameters_json'].val
                    if params_json and params_json != '{}':
                        try:
                            params_dict = json.loads(params_json)
                            is_first_param = True
                            for param_name, param_metadata in params_dict.items():
                                add_section = is_first_param
                                self._create_single_dynamic_parameter(
                                    proc_name, param_name, param_metadata,
                                    section=add_section, page='Fx', prefix='Fx',
                                    order=current_order, context=fx_context
                                )
                                is_first_param = False
                                current_order += 1  # Increment order for each param
                            is_first_processor = False
                        except:
                            pass
                    break

    #endregion Fx DYNAMIC PARAMETERS

    #region DYNAMIC PARAMETER STATE MANAGEMENT - Save/Restore last_val and last_mode

    def _ensure_dyn_param_state_table(self):
        """
        Ensure table_dyn_param_states exists with proper columns.
        Columns: param_name, last_val, last_mode, last_expr
        """
        state_table = self.ownerComp.op('table_dyn_param_states')

        if not state_table:
            self.logger.log('WARNING: table_dyn_param_states not found, cannot save parameter states', level='WARNING')
            return None

        # Check if table has correct headers, if not initialize
        if state_table.numRows == 0 or state_table[0, 0].val != 'param_name':
            state_table.clear()
            state_table.appendRow(['param_name', 'last_val', 'last_mode', 'last_expr'])

        return state_table

    def save_dyn_param_state(self, param_name, context=None):
        """
        Save the current value, mode, and expression of a Dyn* parameter.
        Called before switching parameter modes to preserve the previous state.

        Args:
            param_name: Name of the parameter (e.g., 'Dyncannylowthreshold')
            context: Optional context string (e.g., "latent_noise|feedback_transform" for FX sequence)
        """
        state_table = self._ensure_dyn_param_state_table()
        if not state_table:
            return

        # Get the parameter
        if not hasattr(self.ownerComp.par, param_name):
            return

        par = getattr(self.ownerComp.par, param_name)

        # Get current state
        # Use .val instead of .eval() to avoid evaluating expressions that might reference destroyed parameters
        try:
            current_val = par.val
        except:
            current_val = par.default  # Fall back to default if val fails
        current_mode = str(par.mode)  # 'ParMode.CONSTANT', 'ParMode.EXPRESSION', 'ParMode.BIND'
        current_expr = par.expr if par.mode == ParMode.EXPRESSION else ''

        # Build state key with context if provided
        state_key = f"{context}:{param_name}" if context else param_name

        # Find existing row or create new one
        row_index = None
        for row in range(1, state_table.numRows):
            if state_table[row, 'param_name'].val == state_key:
                row_index = row
                break

        if row_index is None:
            # Create new row
            state_table.appendRow([state_key, str(current_val), current_mode, current_expr])
        else:
            # Update existing row
            state_table[row_index, 'last_val'] = str(current_val)
            state_table[row_index, 'last_mode'] = current_mode
            state_table[row_index, 'last_expr'] = current_expr

        # self.logger.log(f'[STATE SAVE] key="{state_key}" | mode={current_mode} | expr="{current_expr[:50] if current_expr else "NONE"}"', level='DEBUG')

    def restore_dyn_param_state(self, param_name, context=None):
        """
        Restore the last saved value, mode, and expression of a Dyn* parameter.
        Call this when switching back to a previous mode.

        Args:
            param_name: Name of the parameter (e.g., 'Dyncannylowthreshold')
            context: Optional context string (e.g., "latent_noise|feedback_transform" for FX sequence)

        Returns:
            dict with 'val', 'mode', 'expr' or None if no saved state exists
        """
        state_table = self._ensure_dyn_param_state_table()
        if not state_table:
            return None

        # Build state key with context if provided
        state_key = f"{context}:{param_name}" if context else param_name

        # Find saved state
        for row in range(1, state_table.numRows):
            if state_table[row, 'param_name'].val == state_key:
                saved_state = {
                    'val': state_table[row, 'last_val'].val,
                    'mode': state_table[row, 'last_mode'].val,
                    'expr': state_table[row, 'last_expr'].val
                }
                # self.logger.log(f'[STATE RESTORE] key="{state_key}" | mode={saved_state["mode"]} | expr="{saved_state["expr"][:50] if saved_state["expr"] else "NONE"}"', level='DEBUG')
                return saved_state

        # self.logger.log(f'[STATE RESTORE] key="{state_key}" | NOT FOUND', level='DEBUG')
        return None

    def apply_dyn_param_state(self, param_name, state):
        """
        Apply a saved state to a Dyn* parameter.

        Args:
            param_name: Name of the parameter
            state: dict with 'val', 'mode', 'expr' from restore_dyn_param_state()
        """
        if not state or not hasattr(self.ownerComp.par, param_name):
            return

        par = getattr(self.ownerComp.par, param_name)

        # Apply based on mode
        mode_str = state['mode']

        if 'EXPRESSION' in mode_str and state['expr']:
            par.mode = ParMode.EXPRESSION
            par.expr = state['expr']
        elif 'BIND' in mode_str:
            par.mode = ParMode.BIND
            # Note: bind expressions need to be set appropriately
            if state['expr']:
                par.bindExpr = state['expr']
        else:  # CONSTANT or default
            par.mode = ParMode.CONSTANT
            # Try to set value with proper type conversion
            try:
                if par.isFloat:
                    par.val = float(state['val'])
                elif par.isInt:
                    par.val = int(float(state['val']))
                elif par.isToggle:
                    par.val = bool(int(float(state['val'])))
                else:
                    par.val = state['val']
            except:
                self.logger.log(f'WARNING: Could not restore value for {param_name}', level='WARNING')

    def save_all_dyn_param_states(self):
        """
        Save state for ALL Dyn* parameters currently on the component.
        Useful to call before regenerating parameters.
        """
        for par_tuple in self.ownerComp.customPars:
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            if par.name.startswith('Dyn'):
                self.save_dyn_param_state(par.name)

    def _get_processor_stage_from_table(self, processor_name, table_dat):
        """
        Get processor stage from table_preprocessors.
        """
        if not table_dat:
            return 'image_pre'

        for row in range(1, table_dat.numRows):
            if table_dat[row, 'name'].val == processor_name:
                stage_cell = table_dat[row, 'stage']
                if stage_cell and stage_cell.val:
                    return stage_cell.val
                break

        return 'image_pre'

    #endregion DYNAMIC PARAMETER STATE MANAGEMENT

    #region ASYNC HELPERS - Methods using TDAsyncIO for non-blocking operations

    async def async_get_preprocessors(self):
        """
        Async subprocess to discover processors (core + custom) via runtime registry import.
        Returns JSON list of processor metadata.
        """
        base_folder = self.ownerComp.par.Basefolder.eval()
        if not base_folder:
            raise Exception("Basefolder parameter not set")

        # Build the Python command to import registry and extract metadata
        python_cmd = f"""
import sys
import json
import logging

# Suppress INFO logs from processor discovery
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, r'{base_folder}\\src')

try:
    from streamdiffusion.preprocessing.processors import _preprocessor_registry

    # Get list of core processors from static registry
    core_processors = set()
    init_file = r'{base_folder}\\src\\streamdiffusion\\preprocessing\\processors\\__init__.py'
    with open(init_file, 'r') as f:
        content = f.read()
        # Find static registry definition
        import re
        match = re.search(r'_preprocessor_registry\\s*=\\s*{{([^}}]+)}}', content, re.DOTALL)
        if match:
            core_names = re.findall(r'"([^"]+)":', match.group(1))
            core_processors.update(core_names)

    # Also add conditional processors
    if 'DEPTH_TENSORRT_AVAILABLE' in content:
        core_processors.add('depth_tensorrt')
    if 'POSE_TENSORRT_AVAILABLE' in content:
        core_processors.add('pose_tensorrt')
    if 'TEMPORAL_NET_TENSORRT_AVAILABLE' in content:
        core_processors.add('temporal_net_tensorrt')
    if 'MEDIAPIPE_POSE_AVAILABLE' in content:
        core_processors.add('mediapipe_pose')
    if 'MEDIAPIPE_SEGMENTATION_AVAILABLE' in content:
        core_processors.add('mediapipe_segmentation')

    result = []
    for name, proc_class in _preprocessor_registry.items():
        try:
            metadata = proc_class.get_preprocessor_metadata()
            is_custom = name not in core_processors
            result.append({{
                'name': name,
                'display_name': metadata.get('display_name', name),
                'description': metadata.get('description', ''),
                'stage': metadata.get('stage', 'image_pre'),
                'parameters': metadata.get('parameters', {{}}),
                'use_cases': metadata.get('use_cases', []),
                'is_custom': is_custom
            }})
        except Exception as e:
            pass

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""

        # Run subprocess using venv Python with hidden window (Windows)
        venv_python = f"{base_folder}\\venv\\Scripts\\python.exe"

        # Hide command window on Windows to prevent flash
        import sys as sys_module
        if sys_module.platform == 'win32':
            # CREATE_NO_WINDOW flag for Windows
            import subprocess as subprocess_module
            startupinfo = subprocess_module.STARTUPINFO()
            startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess_module.SW_HIDE
        else:
            startupinfo = None

        # Use asyncio.create_subprocess_exec for non-blocking subprocess
        proc = await asyncio.create_subprocess_exec(
            venv_python, '-c', python_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=base_folder,
            startupinfo=startupinfo
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise Exception(f"Subprocess failed: {stderr.decode()}")

        # Parse JSON output (find JSON array, ignoring print statements)
        stdout_str = stdout.decode().strip()
        json_start = stdout_str.find('[')
        if json_start == -1:
            raise Exception(f"No JSON array in output: {stdout_str[:200]}")

        json_str = stdout_str[json_start:]
        processors_data = json.loads(json_str)

        if isinstance(processors_data, dict) and 'error' in processors_data:
            raise Exception(f"Failed to load processors: {processors_data['error']}")

        return processors_data

    def _on_processors_discovered(self, task):
        """
        Callback when async processor discovery completes.
        Populates table_preprocessors with discovered processors.

        Args:
            task: AsyncIOTask object from TDAsyncIO
        """
        try:
            if task.error:
                self.logger.log(f'Processor discovery failed: {str(task.error)}', level='ERROR')
                return

            processors_data = task.result
            if not processors_data:
                self.logger.log('No processors discovered', level='WARNING')
                return

            # Populate table_preprocessors
            table_dat = self.ownerComp.op('table_preprocessors')
            if not table_dat:
                self.logger.log('table_preprocessors DAT not found', level='ERROR')
                return

            import json

            table_dat.clear()
            table_dat.appendRow(['name', 'is_tensorrt', 'is_custom', 'display_name', 'description', 'stage', 'parameters_json', 'use_cases'])

            for proc_data in sorted(processors_data, key=lambda x: x['name']):
                name = proc_data['name']
                is_trt = 'tensorrt' in name.lower() or 'trt' in name.lower()
                is_custom = proc_data.get('is_custom', False)
                display_name = proc_data.get('display_name', name)
                description = proc_data.get('description', '')
                stage = proc_data.get('stage', 'image_pre')

                params_dict = proc_data.get('parameters', {})
                parameters_json = json.dumps(params_dict, indent=2) if params_dict else '{}'

                use_cases_list = proc_data.get('use_cases', [])
                use_cases = ', '.join(use_cases_list) if use_cases_list else ''

                table_dat.appendRow([
                    name,
                    '1' if is_trt else '0',
                    '1' if is_custom else '0',
                    display_name,
                    description,
                    stage,
                    parameters_json,
                    use_cases
                ])

            self.logger.log(f'Discovered {len(processors_data)} processors ({sum(1 for p in processors_data if p.get("is_custom"))} custom)', level='INFO')

        except Exception as e:
            self.logger.log(f'Error in processor discovery callback: {e}', level='ERROR')
            import traceback
            traceback.print_exc()

    #endregion ASYNC HELPERS

    #region DotToxLoader Integration - Package/Unload for Distribution

    def Unloadoperator(self):
        """
        Package this operator for distribution and replace with dotloader stub.

        The result is a lightweight stub that can be shared without actual code.
        Users with the TOX can drop it onto the stub to restore full functionality.
        """
        dotloader = self.ownerComp.op('dotloader')
        if not dotloader:
            ui.messageBox('Error', "Missing 'dotloader' component.", buttons=['OK'])
            return False

        try:
            loader_ext = dotloader.ext.DotToxLoaderExt
        except AttributeError:
            ui.messageBox('Error', "DotToxLoaderExt not found on dotloader.", buttons=['OK'])
            return False

        # Confirm with user
        confirm = ui.messageBox(
            'Unload StreamDiffusionTD',
            'This will package StreamDiffusionTD for distribution:\n\n'
            '• All parameters will be saved\n'
            '• Connections will be preserved\n'
            '• The full operator will be replaced with a lightweight stub\n\n'
            'To restore, drop the StreamDiffusionTD.tox onto the stub.\n\n'
            'Continue?',
            buttons=['Unload', 'Cancel']
        )
        if confirm != 0:
            return False

        # Let DotToxLoaderExt handle everything
        stub = loader_ext.unload_operator(self.ownerComp, 'StreamDiffusionTD')
        return stub is not None

    def _check_and_restore_from_dotloader(self):
        """
        Check if there's pending restore data from a DotToxLoader stub.
        Called during __init__ to restore parameters after TOX load.
        Also saves TOX path to disk for future auto-loading.
        """
        try:
            loader_ext = self.ownerComp.op('dotloader').ext.DotToxLoaderExt
            loader_ext.restore_from_stub(self.ownerComp)
            loader_ext.save_tox_path(operator_type='StreamDiffusionTD')
        except:
            pass


    def _configure_registry(self):
        """Register this operator with the TOX registry via DotToxLoader."""
        try:
            dotloader = self.ownerComp.op('dotloader')
            if not dotloader or not hasattr(dotloader, 'ext'):
                return

            loader_ext = dotloader.ext.DotToxLoaderExt

            # Get version from operator
            version = str(self.ownerComp.par.Txversion.eval()) if hasattr(self.ownerComp.par, 'Txversion') else '0.3.1'

            # Register - registry handles saving the TOX
            loader_ext.configure_registry(
                tox_name=self.TOX_NAME,
                branch=self.BRANCH,
                version=version
            )
        except Exception as e:
            self.logger.log(f'Error configuring registry: {e}', level='WARNING')


    #endregion DotToxLoader Integration

    #region Tox Updater Integration - Dynamic update button with state-based UX

    def _get_tox_updater(self):
        """Get the tox_updater extension."""
        return self.ownerComp.op('tox_updater').ext.ToxUpdaterEXT

    def _init_update_checker(self):
        """Initialize the update checker on startup."""
        self.ownerComp.par.Updateaction.label = 'Check for Updates'
        self.ownerComp.par.Updateaction.enable = True
        run('args[0]()', self._check_update_status, delayFrames=60)

    def _check_update_status(self, force=False):
        """Check auth and update status, update Updateaction parameter."""
        updater = self._get_tox_updater()
        update_par = self.ownerComp.par.Updateaction

        if not updater.IsAuthenticated():
            self.logger.log("Sign in to check for updates [see 'About' page]", level='INFO')
            update_par.label = 'Sign In'
            update_par.enable = True
            return

        # Check if registry data already loaded (from cache or previous fetch)
        # Skip cache on force=True (manual button click)
        if not force and (updater.GetOpInfo(self.TOX_NAME) is not None or updater._operator_menu_items):
            self._on_update_check_complete()
            return

        # Need to fetch from network
        update_par.label = 'Checking...'
        update_par.enable = False

        # Small delay so user sees "Checking..." feedback
        def delayed_complete():
            run('args[0]()', self._on_update_check_complete, delayFrames=20)
        updater.Checkupdates(on_complete=delayed_complete)

    def _on_update_check_complete(self):
        """Callback after registry fetch completes."""
        updater = self._get_tox_updater()
        update_par = self.ownerComp.par.Updateaction

        info = updater.GetOpInfo(self.TOX_NAME)
        if info and info.get('has_update'):
            version = info.get('version', '?')
            current = info.get('installed_version', '?')
            self.logger.log(f"Update available: v{current} → v{version} [see 'About' page]", level='INFO')
            update_par.label = f'Install v{version}'
            update_par.enable = True
            self._pending_update_version = version
        elif info:
            # In registry, no update available
            current = info.get('installed_version', self.ownerComp.par.Txversion.eval())
            self.logger.log(f'StreamDiffusionTD v{current} [latest]', level='INFO')
            update_par.label = 'Up to Date ✓'
            update_par.enable = True
            self._pending_update_version = None
        else:
            # Not in registry yet - show local version
            current = self.ownerComp.par.Txversion.eval()
            self.logger.log(f'StreamDiffusionTD v{current} [not in registry]', level='INFO')
            update_par.label = 'Up to Date ✓'
            update_par.enable = True
            self._pending_update_version = None

    def Updateaction(self):
        """Handle the Updateaction pulse parameter."""
        updater = self._get_tox_updater()

        if not updater.IsAuthenticated():
            self.logger.log("Opening browser for sign in...", level='INFO')
            updater.Signin(on_authenticated=self._check_update_status)
            return

        if self._pending_update_version:
            self._download_update()
        else:
            self._check_update_status(force=True)

    def _download_update(self):
        """Download the latest update using async callback."""
        updater = self._get_tox_updater()
        update_par = self.ownerComp.par.Updateaction

        update_par.label = 'Downloading...'
        update_par.enable = False

        self.logger.log('Downloading StreamDiffusionTD update...', level='INFO')

        updater.DownloadOperatorAsync(
            self.TOX_NAME,
            on_complete=self._on_update_downloaded,
            on_error=self._on_update_error
        )

    def _on_update_downloaded(self, op_name: str, version: str, path: str):
        """Callback when update download completes."""
        self.logger.log(f'Downloaded {op_name} v{version} to {path}', level='INFO')

        update_par = self.ownerComp.par.Updateaction
        update_par.label = 'Downloaded!'
        update_par.enable = False

        result = ui.messageBox(
            'Update Downloaded',
            f'StreamDiffusionTD v{version} downloaded.\n\nReload operator now?\n\nYour settings will be preserved.',
            buttons=['Reload Now', 'Later']
        )

        if result == 0:
            self._apply_update(path)
        else:
            update_par.label = 'Reload Update'
            update_par.enable = True
            self._pending_update_path = path

    def _on_update_error(self, op_name: str, error: str):
        """Callback when update download fails."""
        self.logger.log(f'Update download failed: {error}', level='ERROR')

        update_par = self.ownerComp.par.Updateaction
        update_par.label = 'Download Failed'
        update_par.enable = True

        ui.messageBox('Download Failed', f'Could not download update:\n{error}', buttons=['OK'])
        run('args[0]()', self._check_update_status, delayFrames=60)

    def _apply_update(self, tox_path: str):
        """Apply update via DotToxLoader."""
        dotloader = self.ownerComp.op('dotloader')
        loader_ext = dotloader.ext.DotToxLoaderExt
        self.logger.log(f'Swapping to updated TOX: {tox_path}', level='INFO')
        loader_ext.swap_operator(tox_path)

    #endregion Tox Updater Integration
