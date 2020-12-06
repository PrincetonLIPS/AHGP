import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):
  """
  Linear Module
  """
  def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
    """
    :param in_dim: dimension of input
    :param out_dim: dimension of output
    :param bias: boolean. if True, bias is included.
    :param w_init: str. weight inits with xavier initialization.
    """
    super(Linear, self).__init__()
    self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    nn.init.xavier_uniform_(
      self.linear_layer.weight,
      gain=nn.init.calculate_gain(w_init))

  def forward(self, x):
    return self.linear_layer(x)

class MultiheadAttention(nn.Module):
  """
  Multihead attention mechanism (dot attention)
  """
  def __init__(self, num_hidden_k, dropout_p=0.1):
    """
    :param num_hidden_k: dimension of hidden 
    """
    super(MultiheadAttention, self).__init__()

    self.num_hidden_k = num_hidden_k
    self.attn_dropout = nn.Dropout(p=dropout_p)

  def forward(self, key, value, query, mask=None):
    # Get attention score
    # query, key, value: B x h x N x dv
    attn = torch.matmul(query, key.transpose(2, 3)) # B x h x N x N
    attn = attn / math.sqrt(self.num_hidden_k)
    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)
    
    attn = torch.softmax(attn, dim=-1)
    # Dropout
    attn = self.attn_dropout(attn)
    # Get Context Vector
    result = torch.matmul(attn, value)

    return result, attn

class AttentionLayer(nn.Module):
  """
  Attention layer used in Bert
  """
  def __init__(self, num_hidden, num_intermediate, h=4):
    """
    :param num_hidden: dimension of hidden
    :param h: num of heads 
    """
    super(AttentionLayer, self).__init__()

    self.num_hidden = num_hidden
    self.num_hidden_per_attn = num_hidden // h
    self.h = h

    self.key = Linear(num_hidden, num_hidden, bias=False)
    self.value = Linear(num_hidden, num_hidden, bias=False)
    self.query = Linear(num_hidden, num_hidden, bias=False)

    self.multihead = MultiheadAttention(self.num_hidden_per_attn)

    self.attn_ouput_dropout = nn.Dropout(p=0.1)
    self.final_output_dropout = nn.Dropout(p=0.1)

    self.attn_linear = Linear(num_hidden, num_hidden)
    self.intermedia_linear = Linear(num_hidden, num_intermediate)
    self.final_linear = Linear(num_intermediate, num_hidden)

    self.attention_layer_norm = nn.LayerNorm(num_hidden)
    self.output_layer_norm = nn.LayerNorm(num_hidden)

  def forward(self, key, value, query, mask=None):
    batch_size = key.size(0)
    seq_k = key.size(1)
    seq_q = query.size(1)
    seq_v = value.size(1)
    residual = value
    
    # Make multihead: B x N x h x dv
    key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
    value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
    query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)
    
    # Transpose for attention dot product: B x h x N x dv
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    if mask is not None:
      mask = mask.unsqueeze(1).unsqueeze(1) # B x N --> B x 1 x 1 x N

    # Get context vector
    attn_output, attns = self.multihead(key, value, query, mask=mask)
    # Concatenate all multihead context vector
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_k, -1) # B X N X d   
    # linear layer after attention
    attn_output = self.attn_linear(attn_output)
    attn_output = self.attn_ouput_dropout(attn_output)
    # residual connection and layernorm
    attn_output = self.attention_layer_norm(attn_output+residual)
    # intermediate linear layer and activation
    intermediate_output = F.gelu(self.intermedia_linear(attn_output))
    # Final linear and activation
    final_output = F.gelu(self.final_linear(intermediate_output))
    # Residual dropout & connection
    final_output = self.final_output_dropout(final_output)
    final_output = final_output + attn_output
    # Layer normalization
    final_output = self.output_layer_norm(final_output)

    return final_output, attns



class Attention(nn.Module):
  """
  Attention Layer used in Tranformer
  """
  def __init__(self, num_hidden, h=4):
    """
    :param num_hidden: dimension of hidden
    :param h: num of heads 
    """
    super(Attention, self).__init__()

    self.num_hidden = num_hidden
    self.num_hidden_per_attn = num_hidden // h
    self.h = h

    self.key = Linear(num_hidden, num_hidden, bias=False)
    self.value = Linear(num_hidden, num_hidden, bias=False)
    self.query = Linear(num_hidden, num_hidden, bias=False)

    self.multihead = MultiheadAttention(self.num_hidden_per_attn)

    self.residual_dropout = nn.Dropout(p=0.1)

    self.final_linear = Linear(num_hidden * 2, num_hidden)

    self.layer_norm = nn.LayerNorm(num_hidden)

  def forward(self, key, value, query, mask=None):
    batch_size = key.size(0)
    seq_k = key.size(1)
    seq_q = query.size(1)
    seq_v = value.size(1)
    residual = value
    
    # Make multihead: B x N x h x dv
    key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
    value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
    query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)
    
    # Transpose for attention dot product: B x h x N x dv
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    if mask is not None:
      mask = mask.unsqueeze(1).unsqueeze(1) # B x N --> B x 1 x 1 x N

    # Get context vector
    result, attns = self.multihead(key, value, query, mask=mask)
    # Concatenate all multihead context vector
    result = result.transpose(1, 2).contiguous().view(batch_size, seq_k, -1) # B X N X d
    
    # Concatenate context vector with input (most important)
    result = torch.cat([residual, result], dim=-1)
    
    # Final linear
    result = F.relu(self.final_linear(result))

    # Residual dropout & connection
    result = self.residual_dropout(result)
    result = result + residual
    # Layer normalization
    result = self.layer_norm(result)

    return result, attns