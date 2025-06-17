from copy import deepcopy
import logging
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from .language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import PreTrainedModel
from accelerate import PartialState
from peft import (
        PeftConfig,
        PeftModel,
        PeftModelForCausalLM,
        PeftModelForSeq2SeqLM,
        PromptLearningConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ..import_utils import is_transformers_greater_than
import importlib
import sys
import os

LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]

if is_transformers_greater_than("4.33.0"):
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
else:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

if sys.version_info < (3, 8):
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True
    
    
def is_accelerate_greater_20_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        accelerate_version = version("accelerate")
    else:
        import pkg_resources

        accelerate_version = pkg_resources.get_distribution("accelerate").version
    return accelerate_version >= "0.20.0"


def is_xpu_available() -> bool:
    if is_accelerate_greater_20_0():
        import accelerate

        return accelerate.utils.is_xpu_available()
    else:
        if importlib.util.find_spec("intel_extension_for_pytorch") is None:
            return False
        try:
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except RuntimeError:
            return False

def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None

class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config):
        super().__init__()
        
        summary_dropout_prob = getattr(config, "summary_dropout_prob", 0.1)

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        self.is_sequential_parallel = False
        
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class LlavaLlamaForCausalLMWithValueHead(PreTrainedModel):
    
    
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, model):
        super().__init__(model.config)
        
        self.model = model

        if not any(hasattr(self.model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")
        
        self.config = self.model.config
        self.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.is_loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        self.is_loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        self.is_sequential_parallel = False

        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = self.model.gradient_checkpointing_disable

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = self.model.gradient_checkpointing_enable


        self.v_head = ValueHead(self.model.config)
        
        self.is_peft_model = False

        self.head_init_weights()
        
        self.other_init()
        
    def other_init(self):
        if is_peft_available():
            if isinstance(self.model, PeftModel):
                self.is_peft_model = True
                # for backward compatibility
                if hasattr(self.model, "active_peft_config") and isinstance(
                    self.model.active_peft_config, PromptLearningConfig
                ):
                    raise ValueError("PromptLearningConfig is not supported for PPO training.")
        print(f"is peft model: {self.is_peft_model}")
        self.current_device = self._get_current_device()
        # self.device = self.current_device
        self.post_init(self.model.state_dict())

    def head_init_weights(self):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = 0.2

        self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
        self.v_head.summary.bias.data.zero_()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Applies a forward pass to the llava model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        output_hidden_states = True  # this had already been set in the LORA / PEFT examples

        if self.is_peft_model and self.model.active_peft_config.peft_type == "PREFIX_TUNING":
            past_key_values = None
        base_model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        if last_hidden_state.device != self.v_head.summary.weight.device:
                last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        value = self.v_head(last_hidden_state).squeeze(-1)
        
        b, seq_len, vocab_size = lm_logits.shape
        final_logits = torch.zeros((b, input_ids.shape[1], vocab_size),dtype=lm_logits.dtype,device=lm_logits.device)
        final_value = torch.zeros((b, input_ids.shape[1]),dtype=value.dtype,device=value.device)
        loss = base_model_output.loss
        
        for b, input_id in enumerate(input_ids):
            image_indexs = torch.where(input_id==IMAGE_TOKEN_INDEX)
            text_indexs = torch.where((input_id != IMAGE_TOKEN_INDEX) & (input_id != 0))[0]
            input_indexs = torch.where(input_id!=0)
            text_len = text_indexs.shape[0]

            t_lm_logits = lm_logits[b,:text_len,:]
            t_value = value[b,:text_len]
            i_lm_logits = torch.mean(lm_logits[b,text_len:,:], dim=0)
            i_value = torch.mean(value[b,text_len:])
            final_logits[b, text_indexs, :] = t_lm_logits
            final_value[b, text_indexs] = t_value
            final_logits[b, image_indexs, :] = i_lm_logits
            final_value[b, image_indexs] = i_value
            
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.bfloat16()

        return (final_logits, loss, final_value)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        self.model.initialize_vision_tokenizer(model_args, tokenizer)
    
    def get_model(self):
        return self.model.get_model()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def resize_token_embeddings(self, new_num_tokens: int = None):
        return self.model.resize_token_embeddings(new_num_tokens)
    
    def get_vision_tower(self):
        return self.model.get_vision_tower()
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        return self.model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    
    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.model, "v_head", self.v_head)

        return self.model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict=None):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.model, "hf_device_map"):
            if (
                "cpu" in self.model.hf_device_map.values()
                or "disk" in self.model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
            
    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory. This method is a wrapper around
        `transformers.PreTrainedModel.save_pretrained`. Please refer to the documentation
        of `transformers.PreTrainedModel.save_pretrained` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        # if it is a peft model only save the `v_head` state_dict and
        # pop the `state_dict` from the kwargs to avoid slient bugs with `peft`
        if self.is_peft_model:
            save_path = args[0]
            save_path = os.path.join(save_path, "pytorch_model.bin")
            torch.save(state_dict, save_path)
            _ = kwargs.pop("state_dict", None)

        return self.model.save_pretrained(*args, **kwargs)
    
    @classmethod
    def _get_current_device(cls):
        r"""
        Get the current device. For GPU, we return the local process index using the `accelerate.PartialState`
        object to handle corner cases when running scripts in distributed environments.

        Returns:
            current_device (`Union[int, str]`):
                The current device.
        """
        state = PartialState()
        if is_xpu_available():
            return f"xpu:{state.local_process_index}"
        else:
            return state.local_process_index if torch.cuda.is_available() else "cpu"
        

def create_reference_model(
    model, num_shared_layers: int = None, pattern: str = None
):
   
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any([pattern_candidate in name for name in parameter_names]):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        ref_param = ref_model.get_parameter(param_name)  # noqa
        ref_param = param  # noqa

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()