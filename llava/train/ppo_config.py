# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import tyro
from typing_extensions import Annotated

from trl.trainer.utils import exact_div

from ..core import flatten_dict
from ..import_utils import is_wandb_available


JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]


@dataclass
class PPOConfig:
    """
    Configuration class for PPOTrainer
    """

    # common parameters
    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    """the name of this experiment (by default is the file name without the extension name)"""
    seed: int = 0
    """Seed value for random generations"""
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    """Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""
    task_name: Optional[str] = None
    """Name of task to use - used only for tracking purposes"""
    model_name: Optional[str] = None
    """Name of model to use - used only for tracking purposes"""
    query_dataset: Optional[str] = None
    """Name of dataset to query - used only for tracking purposes"""
    reward_model: Optional[str] = None
    """The reward model to use - used only for tracking purposes"""
    remove_unused_columns: bool = True
    """Remove unused columns from the dataset if `datasets.Dataset` is used"""
    tracker_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the tracker (e.g. python ppo.py --ppo_config.tracker_kwargs='{"wandb": {"entity": "my_wandb_entity", "name": "my_exp_name"}}'"""
    accelerator_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator"""
    project_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator project config (e.g. `logging_dir`)"""
    tracker_project_name: str = "trl"
    """Name of project to use for tracking"""
    push_to_hub_if_best_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for pushing model to the hub during training (e.g. repo_id)"""

    # hyperparameters
    steps: int = 20000
    """Number of training steps"""
    learning_rate: float = 1e-5
    """Adam learning rate"""
    adap_kl_ctrl: bool = True
    """Use adaptive KL control, otherwise linear"""
    init_kl_coef: Optional[float] = 0.2
    """Initial KL penalty coefficient (used for adaptive and linear control)"""
    kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl"
    """kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"""
    target: Optional[float] = 6
    """Target KL value for adaptive KL control"""
    horizon: Optional[float] = 10000
    """Horizon for adaptive KL control"""
    gamma: float = 0.99
    """Gamma parameter for advantage calculation"""
    lam: float = 0.95
    """Lambda parameter for advantage calculation"""
    cliprange: float = 0.2
    """Range for clipping in PPO policy gradient loss"""
    cliprange_value: float = 0.2
    """Range for clipping values in loss calculation"""
    vf_coef: float = 0.1
    """Scaling factor for value loss"""
    batch_size: int = 16
    """Number of samples per optimisation step"""
    forward_batch_size: Optional[int] = None
    """DEPRECATED: use `mini_batch_size` instead, which does the same thing."""
    mini_batch_size: int = 1
    """Number of samples optimized in each mini batch"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    world_size: tyro.conf.Suppress[int] = None
    """The world size for distributed training"""
    ppo_epochs: int = 4
    """Number of optimisation epochs per batch of samples"""
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping"""
    optimize_cuda_cache: Optional[bool] = None
    """DEPRECATED: use `optimize_device_cache` instead, which does the same thing."""
    optimize_device_cache: Optional[bool] = False
    """Optimize device cache for slightly more memory-efficient training"""
    early_stopping: bool = False
    """Whether to stop the PPO optimization loop early is the KL too high"""
    target_kl: float = 1
    """Stop early if we exceed this value by over 50%"""
    compare_steps: int = 1
    """Number of steps between comparison of the current reward with the best seen so far"""
    ratio_threshold: float = 10.0
    """Skip mini-batches with high PPO ratios that can cause loss spikes"""
    use_score_scaling: bool = False
    """Use score scaling"""
    use_score_norm: bool = False
    """Use score normalization. Only applicable if use_score_scaling is True"""
    score_clip: Optional[float] = None
    """Score clipping"""
    whiten_rewards: bool = False
    """Whiten the rewards before compute advantages"""

    # computed hyperparameters at runtime; we use `tyro.conf.Suppress` to hide them from the help text
    is_encoder_decoder: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is an encoder-decoder model"""
    is_peft_model: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is a PEFT model"""
    backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: Number of samples optimized in an `optimizer.step()` call"""
    global_backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `backward_batch_size` across all processes"""
    global_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `batch_size` across all processes"""

    if optimize_cuda_cache is not None:
        warnings.warn(
            "The `optimize_cuda_cache` argument will be deprecated soon, please use `optimize_device_cache` instead."
        )
        optimize_device_cache = optimize_cuda_cache
    else:
        optimize_device_cache = False

    def __post_init__(self):
        if self.forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for PPO optimization."
            )
            self.mini_batch_size = self.forward_batch_size

        self.backward_batch_size = self.mini_batch_size * self.gradient_accumulation_steps
        exact_div(
            self.batch_size,
            self.backward_batch_size,
            "`batch_size`",
            "`mini_batch_size * gradient_accumulation_steps`",
            "`batch_size` must be a multiple of `mini_batch_size * gradient_accumulation_steps`",
        )

        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            if not is_wandb_available():
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)