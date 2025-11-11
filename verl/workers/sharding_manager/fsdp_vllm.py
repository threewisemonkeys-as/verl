# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
from typing import Union
import os
import time
from collections import OrderedDict
from typing import List

import torch
from peft import PeftModel
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from vllm import AsyncLLMEngine

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.model import check_exclude_modules, check_target_modules
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class FSDPVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            try:
                self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            except:
                self.model_runner = self.inference_engine.engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            VLLMHijack.hijack()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        def __collect_lora_params()->OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    lora_params = layered_summon_lora_params(self.module)
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                        for name, param in lora_params.items()}
                        else:
                            model = self.module._fsdp_wrapped_module.base_model.model
                            orig_dev = 'cpu' if 'cpu' in str(next(model.parameters()).device) else 'cuda'
                            model = model.to('cpu')
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ['_flat_param', 'lora_']):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                            # model = model.to(orig_dev)
                    torch.cuda.empty_cache()
            else:
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model
                    orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'
                    model = model.to('cpu')
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            continue
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        lora_params[name] = param.detach().cpu()
                    # model = model.to(orig_dev)
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
            params = __collect_lora_params()
        else:
            params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        """
        Robust weight loader for Qwen3 under vLLM:
          - Normalizes HF/PEFT names
          - Adds .base_layer for LoRA when base isn't preloaded
          - Fuses q/k/v -> qkv
          - **Mirrors namespaces to match the *actual* vLLM module tree**:
               root.*, model.*, transformer.* variants as needed, based on hasattr checks.
        """
        model = self.model_runner.model

        # Fast path: base already synced, LoRA-only delta -> use vLLM LoRA API.
        if peft_config and self.base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="in_memory_lora",
                peft_config=asdict(peft_config),
                lora_tensors=updated_params,
            )
            self.inference_engine.llm_engine.add_lora(lora_request)
            logger.info(f"[vLLM] LoRA add via add_lora(); tensors={len(updated_params)}")
            return

        # ------------------------
        # Canonicalize once (same as before)
        # ------------------------
        def _strip_outer_prefixes(k: str) -> str:
            if k.startswith("base_model.model."):
                k = k[len("base_model.model."):]
            # keep inner prefix for now (we will mirror later based on live tree)
            return k

        def _maybe_add_base_layer(k: str, peft_cfg) -> str:
            if ".base_layer." in k or not (k.endswith(".weight") or k.endswith(".bias")):
                return k
            stacked = {"q_proj","k_proj","v_proj","qkv_proj","o_proj","gate_proj","up_proj","down_proj","w1","w2","w3","wi","wo"}
            if any(seg in k for seg in ("embed_tokens","lm_head",".norm.")):
                mod = k.rsplit(".", 1)[0]
                try:
                    if not check_target_modules(peft_cfg, mod):
                        return k
                except Exception:
                    return k
            mod = k.rsplit(".", 1)[0]
            last = mod.split(".")[-1]
            try:
                if check_exclude_modules(peft_cfg, mod):
                    return k
            except Exception:
                pass
            targeted = False
            try:
                targeted = check_target_modules(peft_cfg, mod)
            except Exception:
                pass
            if last in stacked or targeted:
                suffix = ".weight" if k.endswith(".weight") else ".bias"
                return f"{mod}.base_layer{suffix}"
            return k

        def _fuse_qkv(params: dict) -> dict:
            out = dict(params)
            from collections import defaultdict
            groups = defaultdict(dict)
            for name in list(out.keys()):
                if ".self_attn.q_proj" in name or ".self_attn.k_proj" in name or ".self_attn.v_proj" in name:
                    kind = "weight" if name.endswith(".weight") else ("bias" if name.endswith(".bias") else None)
                    if kind is None:
                        continue
                    pre, post = name.split(".self_attn.", 1)
                    has_base = ".base_layer." in name
                    proj = "q_proj" if "q_proj" in post else ("k_proj" if "k_proj" in post else "v_proj")
                    groups[(pre, has_base, kind)][proj] = name
            for (pre, has_base, kind), d in list(groups.items()):
                if not all(p in d for p in ("q_proj","k_proj","v_proj")):
                    continue
                qk, kk, vk = d["q_proj"], d["k_proj"], d["v_proj"]
                tgt = f"{pre}.self_attn.qkv_proj"
                if has_base:
                    tgt += ".base_layer"
                tgt += f".{kind}"
                try:
                    fused = torch.cat([out[qk], out[kk], out[vk]], dim=0)
                except Exception as e:
                    logger.error(f"[vLLM fuse_qkv] concat failed for {pre} ({kind}): {e}")
                    continue
                out[tgt] = fused
                del out[qk]; del out[kk]; del out[vk]
            return out

        need_base_layer = bool(peft_config) and not self.base_sync_done
        p = { _strip_outer_prefixes(k): v for k, v in updated_params.items() }
        if peft_config and need_base_layer:
            p = { _maybe_add_base_layer(k, peft_config): v for k, v in p.items() }
        p = _fuse_qkv(p)

        # ------------------------
        # Namespace mirroring based on the *live* vLLM model structure
        # ------------------------
        def _mirror_namespace(params: dict) -> dict:
            out = dict(params)

            has_model = hasattr(model, "model")
            has_transformer = hasattr(model, "transformer")
            has_root_embed = hasattr(model, "embed_tokens")
            has_model_embed = hasattr(model.model, "embed_tokens") if has_model else False
            has_tx_embed = hasattr(model.transformer, "embed_tokens") if has_transformer else False

            has_model_layers = hasattr(model.model, "layers") if has_model else False
            has_root_layers = hasattr(model, "layers")
            has_tx_layers = hasattr(model.transformer, "layers") if has_transformer else False

            def add_variant(dst_key, src_val):
                if dst_key not in out:
                    out[dst_key] = src_val

            for k, v in list(params.items()):
                # 1) embed_tokens.*
                if k.startswith("embed_tokens."):
                    if has_model_embed:
                        add_variant("model."+k, v)
                    if has_tx_embed:
                        add_variant("transformer."+k, v)
                elif k.startswith("model.embed_tokens.") and has_root_embed:
                    add_variant(k[len("model."):], v)
                elif k.startswith("transformer.embed_tokens.") and has_root_embed:
                    add_variant(k[len("transformer."):], v)

                # 2) layers.*
                if k.startswith("layers."):
                    if has_model_layers:
                        add_variant("model."+k, v)
                    if has_tx_layers:
                        add_variant("transformer."+k, v)
                elif k.startswith("model.layers.") and has_root_layers:
                    add_variant(k[len("model."):], v)
                elif k.startswith("transformer.layers.") and has_root_layers:
                    add_variant(k[len("transformer."):], v)

                # 3) lm_head.* (often root-level in vLLM Qwen)
                if k.startswith("lm_head.") and has_model and hasattr(model, "lm_head") and hasattr(model.model, "lm_head"):
                    add_variant("model."+k, v)
                elif k.startswith("model.lm_head.") and hasattr(model, "lm_head"):
                    add_variant(k[len("model."):], v)

            # Final hint: if model has .model.*, prefer providing those variants
            # (vLLM Qwen loaders often traverse under .model)
            return out

        canon = _mirror_namespace(p)

        # Diagnostics: check which variant exists on the live model
        probe = []
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            probe += ["model.embed_tokens.weight"]
        elif hasattr(model, "embed_tokens"):
            probe += ["embed_tokens.weight"]

        # Check one attn projection and lm_head variants
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            probe += ["model.layers.0.self_attn.o_proj.weight",
                      "model.layers.0.self_attn.qkv_proj.weight"]
        elif hasattr(model, "layers"):
            probe += ["layers.0.self_attn.o_proj.weight",
                      "layers.0.self_attn.qkv_proj.weight"]

        if hasattr(model, "lm_head"):
            probe += ["lm_head.weight"]
        elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
            probe += ["model.lm_head.weight"]

        missing = [k for k in probe if k not in canon]
        if missing:
            logger.warning(f"[vLLM canon/ns] Missing after namespace mirror: {missing}")
        logger.info(f"[vLLM canon/ns] total_keys={len(canon)} "
                    f"tree(has_model={hasattr(model,'model')}, "
                    f"has_root_embed={hasattr(model,'embed_tokens')}, "
                    f"has_model_embed={hasattr(getattr(model,'model',None),'embed_tokens')})")

        # Load
        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()
        loaded_params = model.load_weights(
            (
                (
                    name,
                    (
                        param.to(device, non_blocking=True).full_tensor()
                        if isinstance(param, DTensor)
                        else param.to(device, non_blocking=True)
                    ),
                )
                for name, param in canon.items()
            )
        )

        self.base_sync_done = True
        logger.info(f"[vLLM] load_weights done; loaded_params={len(loaded_params) if loaded_params else -1}")
