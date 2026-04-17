from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class TimeLLM(_BaseNFForecaster):

    _tags = {
        # "ignores-exogeneous-X": True,
        "capability:exogenous": False,
    }

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            patch_len: int = 16,
            stride: int = 8,
            d_ff: int = 128,
            top_k: int = 5,
            d_llm: int = 768,
            d_model: int = 32,
            n_heads: int = 8,
            enc_in: int = 7,
            dec_in: int = 7,
            llm=None,
            llm_config=None,
            llm_tokenizer=None,
            llm_num_hidden_layers=32,
            llm_output_attention: bool = True,
            llm_output_hidden_states: bool = True,
            prompt_prefix: Optional[str] = None,
            dropout: float = 0.1,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            loss="mae",
            valid_loss=None,
            learning_rate: float = 1e-4,
            max_steps: int = 5,
            val_check_steps: int = 100,
            batch_size: int = 32,
            valid_batch_size: Optional[int] = None,
            windows_batch_size: int = 1024,
            inference_windows_batch_size: int = 1024,
            start_padding_enabled: bool = False,
            training_data_availability_threshold=0.0,
            step_size: int = 1,
            num_lr_decays: int = 0,
            early_stop_patience_steps: int = -1,
            scaler_type: str = "identity",
            random_seed: int = 1,
            drop_last_loader: bool = False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            **trainer_kwargs,
    ):
        super().__init__(nfm.TimeLLM, locals())
        return
