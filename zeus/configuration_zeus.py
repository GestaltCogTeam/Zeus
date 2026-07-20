from transformers import PretrainedConfig
from typing import List
import torch


class ZeusConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: List[int] = [256, 256, 512, 512, 512, 256, 256],
        n_heads: List[int] = [4, 4, 8, 8, 8, 4, 4],
        intermediate_size: List[int] = [1024, 1024, 2048, 2048, 2048, 1024, 1024],
        dropout: float = 0.1,
        hidden_act: str = "silu",
        num_reg_tokens: int = 4,
        num_layers: List[int] = [1, 1, 1, 1, 1, 1, 1],
        scales: List[int] = [1, 4, 16, 64, 16, 4, 1],
        quantiles: List[int] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        initializer_range: float = 0.02,
        num_latent_tokens: int = 2,
        attn_implementation: str = "eager",
        use_latent_tokens: bool = True,
        use_pcl: bool = True,
        contrastive_size: int = 768,
        contrastive_loss_weight: float = 1.0,
        num_prototypes: int = 16384,
        student_temp: float = 0.1,
        teacher_temp: float = 0.05,
        temperature: float = 0.1,
        center_momentum: float = 0.9,
        sinkhorn_iters: int = 3,
        sinkhorn_eps: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.scales = scales
        self.quantiles = quantiles
        self.initializer_range = initializer_range
        self.num_reg_tokens = num_reg_tokens
        self.num_latent_tokens = num_latent_tokens
        self.num_layers = num_layers
        self.attn_implementation = attn_implementation
        self.dtype = torch.bfloat16
        self.use_latent_tokens = use_latent_tokens

        self.use_pcl= use_pcl
        self.contrastive_size = contrastive_size
        self.contrastive_loss_weight = contrastive_loss_weight
        self.num_prototypes = num_prototypes
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.center_momentum = center_momentum
