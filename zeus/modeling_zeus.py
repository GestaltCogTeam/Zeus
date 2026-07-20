import numpy as np
import torch
from torch import nn
import math
from typing import Tuple, Optional
from transformers import PreTrainedModel
from .configuration_zeus import ZeusConfig
from basicts.modules import ACT2FN
from basicts.modules.transformer import DecoderOnlyLayer, MultiHeadAttention, RotaryPositionEmbedding, AutoRegressiveDecoder
from basicts.modules.norm import RMSNorm
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input


class ZeusFlashAttention(nn.Module):
    """
    Encoder-only (BERT-style) Multi-Head Attention with FlashAttention v2
    """
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        kv_heads: Optional[int] = None,
        rope: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout_p = dropout
        self.rope = rope

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        # [B, L, H*D] -> [B, L, H, D]
        return x.view(B, L, self.n_heads, self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[object] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        layer_idx: Optional[int] = None,
    ):
        assert not output_attentions, \
            "FlashAttention v2 does not support returning attention weights efficiently."

        B, L, _ = hidden_states.shape
        device = hidden_states.device

        q = self._shape(self.q_proj(hidden_states), B, L)
        k = self._shape(self.k_proj(hidden_states), B, L)
        v = self._shape(self.v_proj(hidden_states), B, L)

        if attention_mask is None:
            mask = torch.ones((B, L), device=device, dtype=torch.bool)

        q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q, attention_mask)
        k_unpad, _, _, _, _ = unpad_input(k, attention_mask)
        v_unpad, _, _, _, _ = unpad_input(v, attention_mask)

        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            pos = position_ids.reshape(-1)[indices]
            q_unpad, k_unpad = self.rope(q_unpad, k_unpad, pos)

        dropout_p = self.dropout_p if self.training else 0.0

        attn_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
            causal=False,
        )

        attn_unpad = attn_unpad.reshape(-1, self.hidden_size)
        context = pad_input(attn_unpad, indices, B, L)

        output = self.out_proj(context)

        return output, None, past_key_value


class ZeusMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class ZeusInputEmbedding(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, hidden_act: str = "gelu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = 4 * self.hidden_size
        self.res_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        return self.res_proj(x) + \
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(DecoderOnlyLayer):
    def __init__(self, config: ZeusConfig, stage: int):
        
        attn_cls = ZeusFlashAttention \
            if config.attn_implementation == "flash_attention_2" else MultiHeadAttention
        
        self_attn = attn_cls(
            hidden_size=config.hidden_size[stage],
            n_heads=config.n_heads[stage],
            dropout=config.dropout,
            rope=RotaryPositionEmbedding(
                dim=config.hidden_size[stage] // config.n_heads[stage],
                max_position_embeddings=4096
            )
        )
        ffn_layer = ZeusMLP(
            config.hidden_size[stage],
            config.intermediate_size[stage],
            config.hidden_act
        )
        super().__init__(self_attn, ffn_layer, (RMSNorm, config.hidden_size[stage]))


class ZeusEncoder(AutoRegressiveDecoder):
    def __init__(self, config: ZeusConfig, stage: int):
        
        decoder_layers = nn.ModuleList(
            [
                EncoderLayer(config, stage)
                for _ in range(config.num_layers[stage])
            ]  
        )

        layer_norm = RMSNorm(config.hidden_size[stage])
        super().__init__(decoder_layers, layer_norm)

        self.num_reg_tokens = config.num_reg_tokens

        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(
                torch.randn(
                    1, self.num_reg_tokens, config.hidden_size[stage]
                ) * config.initializer_range
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs
        ):

        B, L, _ = hidden_states.size()
        position_ids = torch.arange(
            L,
            dtype=torch.long,
            device=hidden_states.device
        ).unsqueeze(0)

        if self.num_reg_tokens > 0:
            reg_tokens = self.reg_tokens.expand(B, -1, -1)
            hidden_states = torch.cat(
                [reg_tokens, hidden_states], dim=1
            )
            position_ids = torch.cat(
                [torch.zeros(
                    1, self.num_reg_tokens,
                    dtype=torch.long,
                    device=hidden_states.device
                ), position_ids], dim=1
            )

        hidden_states, attn_weights, kv_cache = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids.expand(B, -1),
            **kwargs
            )
        
        reg_tokens = hidden_states[:, :self.num_reg_tokens]
        hidden_states = hidden_states[:, self.num_reg_tokens:]
        
        return hidden_states, attn_weights, kv_cache, reg_tokens

class ZeusPoolingLayer(nn.Module):

    def __init__(self, config: ZeusConfig, stage: int):
        super().__init__()
        self.stage = stage
        self.config = config
        self.factor = config.scales[stage] // config.scales[stage - 1]
        self.proj = nn.Linear(
            self.factor * config.hidden_size[stage - 1],
            config.hidden_size[stage],
            bias=False
        )
    
    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor):
        batch_size, _, hidden_size = hidden_states.size()
        hidden_states = hidden_states.reshape(batch_size, -1, self.factor * hidden_size)
        hidden_states = self.proj(hidden_states)
        padding_mask = padding_mask.reshape(batch_size, -1, self.factor, 1).any(dim=2)
        return hidden_states, padding_mask

class ZeusUnpoolingLayer(nn.Module):

    def __init__(self, config: ZeusConfig, stage: int):
        super().__init__()
        self.stage = stage
        self.config = config
        self.factor = config.scales[stage - 1] // config.scales[stage]
        self.proj = nn.Linear(
            config.hidden_size[stage - 1],
            self.factor * config.hidden_size[stage],
            bias=False
        )
    
    def forward(self, hidden_states: torch.Tensor, skip_connection: torch.Tensor):
        batch_size, _, hidden_size = skip_connection.size()
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, -1, hidden_size)
        hidden_states = hidden_states + skip_connection
        return hidden_states

class ZeusPreTrainedModel(PreTrainedModel):
    config_class = ZeusConfig

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Zeus(ZeusPreTrainedModel):

    _supports_flash_attn_2 = True
    
    def __init__(self, config: ZeusConfig):
        super().__init__(config)
        self.config = config
        self.scales = config.scales
        self.num_reg_tokens = config.num_reg_tokens
        self.num_scales = len(self.scales)

        self.input_mlp = ZeusInputEmbedding(
            config.input_dim,
            config.hidden_size[0],
            config.hidden_act
            )

        self.special_tokens = nn.Embedding(2, config.hidden_size[0])
        self.pad_token_id = 0
        self.mask_token_id = 1

        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # first layer
        self.encoders.append(ZeusEncoder(config, 0))

        # down samplers
        for i in range(1, self.num_scales // 2 + 1):
            self.encoders.append(ZeusEncoder(config, i))
            self.downsamplers.append(ZeusPoolingLayer(config, i))
        
        for i in range(self.num_scales // 2 + 1, self.num_scales):
            self.encoders.append(ZeusEncoder(config, i))
            self.upsamplers.append(ZeusUnpoolingLayer(config, i))

        self.num_quantiles = len(config.quantiles)
        quantiles = torch.tensor(config.quantiles)
        self.register_buffer("quantiles", quantiles, persistent=False)
        self.head = nn.Linear(config.hidden_size[-1], self.num_quantiles)

        self.post_init()

    def _prepare_embedding(
            self,
            inputs: torch.Tensor,
            targets_mask: torch.Tensor,
            padding_mask: torch.Tensor = None,
        ):

        B, L, _ = inputs.shape
        input_embeds = self.input_mlp(inputs) # [B, L, D]
        
        is_target = targets_mask == 1
        input_embeds = torch.where(
            is_target,
            self.special_tokens(
                torch.full_like(targets_mask.squeeze(-1), self.mask_token_id)
            ),
            input_embeds)
        
        if padding_mask is not None:
            is_padding = padding_mask == 0
            input_embeds = torch.where(
                is_padding,
                self.special_tokens(
                    torch.full_like(padding_mask.squeeze(-1), self.pad_token_id)
                ),
                input_embeds)
        if padding_mask is None:
            padding_mask = torch.ones(
                (B, L, 1), device=input_embeds.device, dtype=torch.long)
        
        # pad
        max_scale = max(self.scales)
        pad_len = math.ceil(L / max_scale) * max_scale - L
        if pad_len > 0:
            pad_tokens = self.special_tokens(
                torch.full(
                    (B, pad_len),
                    self.pad_token_id,
                    device=input_embeds.device
                )
            )
            
            input_embeds = torch.cat(
                [input_embeds, pad_tokens],dim=1)
            
            padding_mask = torch.cat(
                [
                    padding_mask,
                    torch.zeros(
                        (B, pad_len, 1),
                        device=input_embeds.device,
                        dtype=padding_mask.dtype)
                ],
                dim=1
            )
            
        return input_embeds, padding_mask
    
    def _prepare_attn_mask(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        device = hidden_states.device
        B, L, _ = hidden_states.shape
        
        if padding_mask is None:
            padding_mask = torch.ones(
                (B, L, 1), device=device, dtype=torch.long)

        # reg tokens
        if self.num_reg_tokens > 0:
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (B, self.num_reg_tokens, 1),
                        device=device,
                        dtype=padding_mask.dtype
                    ),
                    padding_mask
                ],
                dim=1
            )
        else:
            attention_mask = padding_mask
        
        if self.config.attn_implementation == "eager":
            attention_mask = attention_mask.view(B, 1, 1, -1) # [B, 1, 1, L]
            attention_mask = (1 - attention_mask.float()) * torch.finfo(hidden_states.dtype).min
        else:
            attention_mask = attention_mask.squeeze(-1) # [B, L]
        return attention_mask
    
    def forward(
        self, 
        inputs: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ):
        """
        x: [B, L, 1]
        padding_mask: [B, L, 1] (0 for padding, 1 for valid)
        target_mask: [B, L, 1] (1 for target/predict, 0 for context)
        """

        # embedding
        ori_seq_len = inputs.shape[1]
        ori_padding_mask = padding_mask
        hidden_states, padding_mask = self._prepare_embedding(inputs, targets_mask, padding_mask)

        scale_outputs = []
        scale_padding_masks = []
        all_hidden_states = []
        reg_token_emb = None
        
        for i in range(self.num_scales):
            
            if i > 0:
                
                # pooling
                if i <= self.num_scales // 2:
                    scale_padding_masks.append(padding_mask)
                    hidden_states, padding_mask = self.downsamplers[i - 1](hidden_states, padding_mask)
            
                # unpooling
                else: # i > self.num_scales // 2
                    idx = i - self.num_scales // 2 - 1
                    hidden_states = self.upsamplers[idx](hidden_states, scale_outputs[self.num_scales - i - 1])
                    padding_mask = scale_padding_masks[self.num_scales - i - 1]
            
            attention_mask = self._prepare_attn_mask(hidden_states, padding_mask)

            hidden_states, _, _, reg_tokens = self.encoders[i](
                hidden_states,
                attention_mask=attention_mask
                )
            
            if i == self.num_scales - 2:
                reg_token_emb = reg_tokens.mean(dim=1) 
            
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)

            if i < self.num_scales:
                scale_outputs.append(hidden_states)

        # [B, L, D] -> [B, L, Q]
        quantile_preds = self.head(hidden_states)[:, :ori_seq_len, :]

        loss = 0.0
        # target and not nan
        if targets is not None:
            loss_mask = (targets_mask * ori_padding_mask).float()
            quantiles = self.quantiles.view(1, 1, self.num_quantiles).to(quantile_preds.dtype)
            loss = 2 * torch.abs((targets - quantile_preds)
                        * ((targets <= quantile_preds).float() - quantiles))
            loss = loss * loss_mask
            loss = loss.sum() / (loss_mask.sum() * self.num_quantiles)

        return {
            "prediction": quantile_preds,
            "loss": loss,
            "all_hidden_states": all_hidden_states,
            "reg_token_emb": reg_token_emb,
        }


class ZeusForPrediction(Zeus):

    def __init__(self, config: ZeusConfig):
        super().__init__(config)

    def generate(
            self,
            context: torch.Tensor,
            prediction_length: int,
            context_mask: torch.Tensor = None,
            use_norm: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:

        context = context.to(self.device)
        
        ndim = context.ndim
        num_features = None
        if ndim == 2:
            context = context.unsqueeze(-1)
        elif ndim == 3 and context.shape[2] > 1:
            _, L, num_features = context.shape
            context = context.transpose(1, 2).view(-1, L, 1)
        elif ndim == 1:
            context = context.unsqueeze(0).unsqueeze(2)
    
        B, L, _ = context.shape
        device = context.device

        if use_norm:
            mean = context.mean(dim=1, keepdim=True)
            std = context.std(dim=1, keepdim=True)
            context = (context - mean) / std
            context = torch.arcsinh(context)
        
        inputs = torch.cat(
            [context, torch.zeros(B, prediction_length, 1, device=device)], dim=1)
        if context_mask is None:
            context_mask = torch.torch.ones(B, L, 1, device=device, dtype=torch.int32)
        padding_mask = torch.cat(
            [
                context_mask,
                torch.ones(B, prediction_length, 1, dtype=torch.int32, device=device)
            ], dim=1
        )
        targets_mask = torch.cat(
            [
                torch.zeros_like(context, dtype=torch.int32),
                torch.ones(B, prediction_length, 1, dtype=torch.int32, device=device)
            ], dim=1
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.forward(
                inputs,
                padding_mask=padding_mask,
                targets_mask=targets_mask,
            )

        # [B, L, Q]
        quantile_preds = outputs["prediction"][:, -prediction_length:, :]

        if use_norm:
            quantile_preds = torch.sinh(quantile_preds)
            quantile_preds = quantile_preds * std + mean

        # [B, L, 1]
        prediction = quantile_preds.mean(dim=-1, keepdim=True)

        if ndim == 2: # [B, L]
            prediction = prediction.squeeze(-1)
        elif ndim == 3 and num_features is not None:
            # [B, L, N]
            prediction = prediction.reshape(-1, num_features, prediction_length).transpose(1, 2)
            prediction = quantile_preds.reshape(
                -1, num_features, prediction_length, quantile_preds.shape[-1]
                ).transpose(1, 2) # [B, L, N, Q]
        elif ndim == 1:
            prediction = prediction[0, :, 0] #[L,]
            quantile_preds = quantile_preds[0] # [L, Q]

        return prediction, quantile_preds
    
    def predict(
            self,
            context,
            prediction_length,
            use_norm: bool = True,
            max_pred_len: int = 4096
            ):

        B = len(context)

        series = []
        Ns = []
        for x in context:
            if x.ndim == 1:          # [L] -> [1, L]
                x = x[None, :]
            else:                    # [L, N] -> [N, L]
                x = x.T
            series.append(x)
            Ns.append(x.shape[0])

        assert len(set(Ns)) == 1, "All arrays must have same N"
        N = Ns[0]

        padded = []
        target_masks = []
        for x in series:  # x: [N, L]
            N_, L = x.shape
            pad = np.full((N_, prediction_length), np.nan)
            padded.append(np.concatenate([x, pad], axis=1))  # [N, L+F]

            m = np.zeros((N_, L + prediction_length), dtype=bool)
            m[:, L:] = 1
            target_masks.append(m)

        batch = []
        # pad_masks = []
        tgt_masks = []

        for x, tm in zip(padded, target_masks):
            N_, Lf = x.shape
            if Lf >= max_pred_len:
                x = x[:, -max_pred_len:]
                tm = tm[:, -max_pred_len:]
                # pm = np.ones((N_, max_pred_len), dtype=bool)
            else:
                pad_len = max_pred_len - Lf
                x = np.concatenate([x, np.full((N_, pad_len), np.nan)], axis=1)
                tm = np.concatenate([tm, np.zeros((N_, pad_len), bool)], axis=1)
                # pm = np.concatenate([np.ones((N_, Lf)), np.zeros((N_, pad_len))], axis=1)

            batch.append(x)
            tgt_masks.append(tm)
            # pad_masks.append(pm)

        # [B, N, T] -> [B*N, T]
        batch = np.stack(batch).reshape(B * N, max_pred_len, 1)
        tgt_masks = np.stack(tgt_masks).reshape(B * N, max_pred_len, 1)
        # pad_masks = np.stack(pad_masks).reshape(B * N, max_pred_len, 1)
        pad_masks = (
            (~np.isnan(batch))
            | (tgt_masks.astype(bool))
        ).astype(np.int32)

        if use_norm:
            mean = np.nanmean(batch, axis=1, keepdims=True)
            std = np.nanstd(batch, axis=1, keepdims=True)
            mean[np.isnan(mean)] = 0.0
            std[np.isnan(std)] = 1.0
            std[std < 1e-3] = 1.0
            batch_norm = (batch - mean) / std
            batch_norm = np.nan_to_num(batch_norm, nan=0.0)
            batch_norm = np.arcsinh(batch_norm)
        else:
            batch_norm = np.nan_to_num(batch, nan=0.0)

        x = torch.from_numpy(batch_norm).to(self.device).float()  # [B*N, T]
        padding_mask = torch.from_numpy(pad_masks).int().to(self.device)
        targets_mask = torch.from_numpy(tgt_masks).int().to(self.device)

        # prediction: [B*N, T]
        # quantile_prediction: [B*N, T, Q]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.forward(
                x,
                padding_mask=padding_mask,
                targets_mask=targets_mask,
            )
        
        quantile_preds = outputs["prediction"].float().detach().cpu().numpy()  # [B*N, T, Q]
        if use_norm:
            quantile_preds = np.sinh(quantile_preds) * std + mean
        quantile_preds = quantile_preds[tgt_masks.repeat(self.num_quantiles, axis=2)].reshape(B, N, prediction_length, self.num_quantiles)

        preds = quantile_preds.mean(axis=-1)

        if N == 1:
            preds = preds[:, 0, :]
            quantile_preds = quantile_preds[:, 0, :, :]

        return preds, quantile_preds


class ZeusForImputation(Zeus):
    def __init__(self, config: ZeusConfig):
        super().__init__(config)
    
    def generate(
            self,
            inputs: torch.Tensor,
            targets_mask: torch.Tensor,
            use_norm: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:

        # transform inputs and targets_mask to [B * N, L, 1]
        ndim = inputs.ndim
        num_features = None
        if ndim == 2:
            inputs = inputs.unsqueeze(-1)
            targets_mask = targets_mask.unsqueeze(-1)
        elif ndim == 3 and inputs.shape[2] > 1:
            _, L, num_features = inputs.shape
            inputs = inputs.transpose(1, 2).reshape(-1, L, 1)
            targets_mask = targets_mask.transpose(1, 2).reshape(-1, L, 1)
        elif ndim == 1:
            inputs = inputs.unsqueeze(0).unsqueeze(2)
            targets_mask = targets_mask.unsqueeze(0).unsqueeze(2)
        
        if use_norm:
            inputs_mask = ~targets_mask # 1 for valid, 0 for invalid
            valid_count = inputs_mask.sum(dim=1, keepdim=True).clamp_min(1)
            mean = inputs.sum(dim=1, keepdim=True) / valid_count
            inputs = (inputs - mean) * inputs_mask
            std = torch.sqrt(
                (inputs ** 2).sum(dim=1, keepdim=True) / valid_count + 1e-5)
            inputs /= std
            inputs = torch.arcsinh(inputs)

        targets_mask = targets_mask.to(torch.int32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self(inputs, targets_mask)
        quantile_preds = outputs["prediction"]

        if use_norm:
            quantile_preds = torch.sinh(quantile_preds)
            quantile_preds = quantile_preds * std + mean

        if num_features is not None:
            quantile_preds = quantile_preds.reshape(-1, num_features, L, self.config.quantiles).transpose(1, 2)

        prediction = quantile_preds.mean(dim=-1, keepdim=True)
        return prediction, quantile_preds


class ZeusForClassification(Zeus):
    def __init__(self, config: ZeusConfig):
        super().__init__(config)
    
    def generate_one_sample(self, inputs: torch.Tensor, padding_mask: torch.Tensor = None, use_norm: bool = True):
        # transform inputs and targets_mask to [B * N, L, 1]
        B = inputs.shape[0]
        ndim = inputs.ndim
        num_features = None
        if ndim == 2:
            inputs = inputs.unsqueeze(-1)
        elif ndim == 3 and inputs.shape[2] > 1:
            _, L, num_features = inputs.shape
            inputs = inputs.transpose(1, 2).view(-1, L, 1)
        elif ndim == 1:
            inputs = inputs.unsqueeze(0).unsqueeze(2)
        
        if use_norm:
            if padding_mask is None:
                padding_mask = torch.ones_like(inputs, dtype=torch.int32)
            valid_count = padding_mask.sum(dim=1, keepdim=True).clamp_min(1)
            mean = inputs.sum(dim=1, keepdim=True) / valid_count
            inputs = (inputs - mean) * padding_mask
            std = torch.sqrt(
                (inputs ** 2).sum(dim=1, keepdim=True) / valid_count + 1e-5)
            inputs /= std
            inputs = torch.arcsinh(inputs)

        targets_mask = torch.zeros_like(inputs, dtype=torch.int32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self(
                inputs,
                targets_mask=targets_mask,
                padding_mask=padding_mask,
                return_all_hidden_states=True
            )
        all_hidden_states = outputs["all_hidden_states"]

        return all_hidden_states
