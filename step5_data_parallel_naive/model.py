import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import process_group_manager as pgm

def flash_attention(q, k, v, causal=True):
    """
    Standard scaled dot-product attention to replace flash_attn_func.
    q, k, v: [batch_size, num_heads, seq_length, head_dim]
    Returns: [batch_size, seq_length, num_heads, head_dim]
    """
    B, H, T, d = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    if causal:
        # Create a causal mask (upper triangular, excluding the diagonal)
        mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [B, H, T, d]
    return out.transpose(1, 2)   # [B, T, H, d]

def apply_rotary_emb(x, cos, sin, interleaved=False):
    """
    Applies rotary positional embeddings.
    x: tensor of shape [batch_size, seq_length, num_heads, head_dim]
    cos, sin: tensors of shape [seq_length, d_rot] (typically, d_rot = head_dim//2).
    Only the first d_rot components are rotated.
    """
    d_rot = cos.shape[-1]
    x_rot = x[..., :d_rot]

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos and sin to [1, seq_length, 1, d_rot] for broadcasting.
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    x_rotated = x_rot * cos + rotate_half(x_rot) * sin

    if x.shape[-1] > d_rot:
        return torch.cat([x_rotated, x[..., d_rot:]], dim=-1)
    else:
        return x_rotated

def layer_norm_fn(hidden_states, weight, bias, residual=None, eps=1e-5, dropout_p=0.0,
                  prenorm=False, residual_in_fp32=False, is_rms_norm=True, return_dropout_mask=False):
    """
    Simple RMSNorm implementation.
    """
    norm = torch.sqrt(torch.mean(hidden_states ** 2, dim=-1, keepdim=True) + eps)
    output = hidden_states / norm * weight
    if residual is not None:
        output = output + residual
    return output

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                               .float().to('cpu') / head_dim))
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    position = torch.arange(seq_length, device=device).unsqueeze(1).float()
    theta = theta.to(device)
    cos = torch.cos(position * theta).to(dtype).repeat(1, 2)
    sin = torch.sin(position * theta).to(dtype).repeat(1, 2)
    return cos, sin

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, hidden_states, residual=None, dropout_p=0.0, prenorm=False,
                residual_in_fp32=False, return_dropout_mask=False):
        return layer_norm_fn(
            hidden_states,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        # TP (Tensor Parallel) assertions and calculations.
        assert config.num_attention_heads % pgm.process_group_manager.tp_world_size == 0, \
            "num_attention_heads should be divisible by tp world size"
        assert config.num_key_value_heads % pgm.process_group_manager.tp_world_size == 0, \
            "num_key_value_heads should be divisible by tp world size"
        self.num_local_heads = config.num_attention_heads // pgm.process_group_manager.tp_world_size
        self.num_local_kv_heads = config.num_key_value_heads // pgm.process_group_manager.tp_world_size

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_idx = layer_idx
        
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        q = self.q_proj(x)  # [batch_size, seq_length, num_heads*head_dim]
        k = self.k_proj(x)  # [batch_size, seq_length, num_key_values*head_dim]
        v = self.v_proj(x)  # [batch_size, seq_length, num_key_values*head_dim]

        q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim)
        q = apply_rotary_emb(q, cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2], interleaved=False)
        k = apply_rotary_emb(k, cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2], interleaved=False)
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
        v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1,2)  # [batch_size, num_key_values, seq_length, head_dim]
     
        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1)  # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1)  # [batch_size, num_heads, seq_length, head_dim]
        
        # For decoding phase, q might have a different seq_length; use causal masking accordingly.
        causal = True if q.size(2) == k.size(2) else False

        out = flash_attention(q, k, v, causal=causal)  # [batch_size, seq_length, num_heads, head_dim]
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim)
        out = self.out_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # Structure: TritonRMSNorm -> Attention -> Residual -> TritonRMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim=head_dim, base=config.rope_theta)

    def forward(self, x, attention_mask=None, position_ids=None):
        cos, sin = self.cos, self.sin 
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # Sanity checks.
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0 
        
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads 
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)])
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.final_norm = TritonRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits  # [batch_size, seq_length, vocab_size]