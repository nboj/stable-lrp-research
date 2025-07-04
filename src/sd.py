from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.models.attention import BasicTransformerBlock
import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn
import torch.nn.functional as F


def forward_crossup(blocklist:CrossAttnUpBlock2D, hidden_states, time, text_embeddings, res_samples):
  resnets = blocklist.resnets
  upsamplers = blocklist.upsamplers
  attentions = blocklist.attentions

  for i in range(len(resnets)):
    resnet = resnets[i]
    transformer = attentions[i]
    hidden_states = torch.cat([hidden_states, res_samples[-(i + 1)]], dim=1)
    hidden_states = forward_resnetblock(resnet, hidden_states, time)
    hidden_states, weights = forward_transformer(transformer, hidden_states, time, text_embeddings)
  if blocklist.upsamplers is not None:
    for i in range(len(upsamplers)):
      upsampler = upsamplers[i] 
      hidden_states = F.interpolate(hidden_states, scale_factor=2)
      hidden_states = upsampler.conv(hidden_states)
  return hidden_states, weights

def forward_up(blocklist:UpBlock2D, hidden_states, time, text_embeddings, res_samples):
  resnets = blocklist.resnets
  upsamplers = blocklist.upsamplers 
  
  for i in range(len(resnets)):
    resnet = resnets[i] 
    hidden_states = torch.cat([hidden_states, res_samples[-(i + 1)]], dim=1)
    hidden_states = forward_resnetblock(resnet, hidden_states, time) 
  if blocklist.upsamplers is not None:
    for i in range(len(upsamplers)):
      upsampler = upsamplers[i] 
      hidden_states = F.interpolate(hidden_states, scale_factor=2)
      hidden_states = upsampler.conv(hidden_states)
  return hidden_states

def forward_crossmid(blocklist:UNetMidBlock2DCrossAttn, hidden_states, time, text_embeddings):
  res1, res2 = blocklist.resnets
  transformer = blocklist.attentions[0]
  
  hidden_states = forward_resnetblock(res1, hidden_states, time)
  hidden_states, weights = forward_transformer(transformer, hidden_states, time, text_embeddings)
  hidden_states = forward_resnetblock(res2, hidden_states, time)

  return hidden_states, weights

def forward_crossdown(blocklist:CrossAttnDownBlock2D, hidden_states, time, text_embeddings):
  res = ()
  
  for i in range(len(blocklist.resnets)):
    resnet = blocklist.resnets[i]
    transformer = blocklist.attentions[i] 

    hidden_states = forward_resnetblock(resnet, hidden_states, time)
    hidden_states, weights = forward_transformer(transformer, hidden_states, time, text_embeddings)
    
    res += (hidden_states,)

  for b in blocklist.downsamplers:
    hidden_states = b.conv(hidden_states)
  res += (hidden_states,)
  return hidden_states, res, weights

def forward_down(blocklist:DownBlock2D, hidden_states, time, text_embeddings):
  res = ()
  
  for i in range(len(blocklist.resnets)):
    resnet = blocklist.resnets[i]
    hidden_states = forward_resnetblock(resnet, hidden_states, time)
    res += (hidden_states,)
  return hidden_states, res




def forward_resnetblock(blocklist:ResnetBlock2D, input_tensor, temb):
  resnet_block = blocklist
  hidden_states = input_tensor

  hidden_states = resnet_block.norm1(hidden_states)
  hidden_states = resnet_block.nonlinearity(hidden_states)

  hidden_states = resnet_block.conv1(hidden_states)

  temb = resnet_block.nonlinearity(temb)
  temb = resnet_block.time_emb_proj(temb)[:, :, None, None]

  hidden_states = hidden_states + temb
  
  hidden_states = resnet_block.norm2(hidden_states)
  hidden_states = resnet_block.nonlinearity(hidden_states)

  hidden_states = resnet_block.dropout(hidden_states)
  hidden_states = resnet_block.conv2(hidden_states)
  

  if resnet_block.conv_shortcut is not None:
      input_tensor = resnet_block.conv_shortcut(input_tensor)

  output_tensor = (input_tensor + hidden_states) / resnet_block.output_scale_factor
  return output_tensor

def forward_transformer(transformer2D:Transformer2DModel, hidden_states, time, text_embeddings):
  norm =  transformer2D.norm
  proj_in = transformer2D.proj_in

  # in transformer
  transformer = transformer2D.transformer_blocks[0] 
  proj_out = transformer2D.proj_out
  residual = hidden_states
  
  hidden_states = norm(hidden_states)
  hidden_states = proj_in(hidden_states)
  
  batch, _, height, width = hidden_states.shape
  inner_dim = hidden_states.shape[1]
  hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim) 

  hidden_states, weights = forward_attention(transformer, hidden_states, text_embeddings)
  
  hidden_states = (
      hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
  )
  hidden_states = proj_out(hidden_states)
  return hidden_states + residual, weights

def forward_attention(attn:BasicTransformerBlock, hidden_states, text_embeddings):
  norm1 = attn.norm1
  norm2 = attn.norm2
  norm3 = attn.norm3
  attn1 = attn.attn1
  attn2 = attn.attn2
  ff = attn.ff
  weights = []

  norm_hidden_states = norm1(hidden_states)
  attn_output = forward_selfattention(attn1, norm_hidden_states)

  hidden_states = attn_output + hidden_states

  norm_hidden_states = norm2(hidden_states) 
  attn_output = forward_crossattention(attn2, norm_hidden_states, text_embeddings)
  
  hidden_states = attn_output + hidden_states

  norm_hidden_states = norm3(hidden_states)
  ff_output = ff(norm_hidden_states, encoder_hidden_states=text_embeddings)
  hidden_states =  hidden_states + ff_output
  
  return hidden_states, weights

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
from torch import matmul, math
def scaled_dot_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
  # print(query.shape, key.shape, value.shape, '    QKV\n\n\n\n\n\n')
  # L, S = query.size(-2), key.size(-2)

  scale_factor = (1 / math.sqrt(query.size(-1))) if scale is None else scale
  
  # attn_bias = torch.zeros(L, S, dtype=query.dtype, device='cuda')
  # if is_causal:
  #     assert attn_mask is None
  #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
  #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
  #     attn_bias.to(query.dtype)

  # if attn_mask is not None:
  #     if attn_mask.dtype == torch.bool:
  #         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
  #     else:
  #         attn_bias += attn_mask
  attn_weight1 = torch.matmul(query, key.transpose(-2, -1) * scale_factor)
  # attn_weight += attn_bias
  attn_weight2 = torch.softmax(attn_weight1, dim=-1)
  # attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
  output = torch.matmul(attn_weight2, value)
  
  return output

def forward_selfattention(selfattention, hidden_states):
  to_q = selfattention.to_q
  to_k = selfattention.to_k
  to_v = selfattention.to_v
  to_out = selfattention.to_out

  query = to_q(hidden_states)
  key = to_k(hidden_states)
  value = to_v(hidden_states)

  inner_dim = key.shape[-1]
  head_dim = inner_dim // selfattention.heads

  batch_size, sequence_length, _ = (
      hidden_states.shape
  )
  query = query.view(batch_size, -1, selfattention.heads, head_dim).transpose(1, 2)
  key = key.view(batch_size, -1, selfattention.heads, head_dim).transpose(1, 2)
  value = value.view(batch_size, -1, selfattention.heads, head_dim).transpose(1, 2)
  
  # hidden_states = F.scaled_dot_product_attention(
  #     query, key, value, dropout_p=0.0, is_causal=False
  # )
  hidden_states = scaled_dot_attention(query, key, value, dropout_p=0.0, is_causal=False) 
  hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, selfattention.heads * head_dim)
  hidden_states = hidden_states.to(query.dtype)

  hidden_states = selfattention.to_out[0](hidden_states)
  hidden_states = selfattention.to_out[1](hidden_states)
  hidden_states = hidden_states / selfattention.rescale_output_factor
  return hidden_states


def forward_crossattention(crossattention, hidden_states, text_embeddings):
  to_q = crossattention.to_q
  to_k = crossattention.to_k
  to_v = crossattention.to_v
  to_out = crossattention.to_out

  query = to_q(hidden_states)
  key = to_k(text_embeddings)
  value = to_v(text_embeddings)

  # print(query.shape, key.shape,value.shape)
  inner_dim = key.shape[-1]
  head_dim = inner_dim // crossattention.heads

  batch_size, sequence_length, _ = (
      text_embeddings.shape
  )

  query = query.view(batch_size, -1, crossattention.heads, head_dim).transpose(1, 2)
  key = key.view(batch_size, -1, crossattention.heads, head_dim).transpose(1, 2)
  value = value.view(batch_size, -1, crossattention.heads, head_dim).transpose(1, 2)
  
  # hidden_states = F.scaled_dot_product_attention(
  #     query, key, value, dropout_p=0.0, is_causal=False
  # )
  hidden_states = scaled_dot_attention(query, key, value, dropout_p=0.0, is_causal=False) 
  hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, crossattention.heads * head_dim)
  hidden_states = hidden_states.to(query.dtype)

  hidden_states = crossattention.to_out[0](hidden_states)
  hidden_states = crossattention.to_out[1](hidden_states)
  hidden_states = hidden_states / crossattention.rescale_output_factor
  return hidden_states


def forward_feedforward(ff, hidden_states):
  net = ff.net
  # FeedForward
  hidden_states_ff = net[0](hidden_states)  # GEGLU
  hidden_states_ff = net[1](hidden_states_ff)  # Dropout
  hidden_states_ff = net[2](hidden_states_ff)  # Linear
  hidden_states = hidden_states + hidden_states_ff  # Residual connection
  return hidden_states

def forward_unet(unet, latents, t, text_embeddings):
  conv_in = unet.conv_in
  time_proj = unet.time_proj
  time_embedding = unet.time_embedding

  down_blocks = unet.down_blocks
  mid_blocks = unet.mid_block
  up_blocks = unet.up_blocks

  conv_norm_out = unet.conv_norm_out
  conv_act = unet.conv_act
  conv_out = unet.conv_out

  with torch.no_grad():
    dtype = unet.dtype
    time = t.expand(latents.shape[0]).cuda()
    time = time_proj(time).to(dtype)
    time = time_embedding(time)

    hidden_states = conv_in(latents)
    samples = (hidden_states,)

    down_layers = down_blocks
    mid_layers = [mid_blocks]
    up_layers = up_blocks
    out_layers = [conv_norm_out, conv_out]
    weights = []

    for i in range(len(down_layers)):
      layer = down_layers[i]
      # print('layer', type(layer))
      if isinstance(layer, CrossAttnDownBlock2D):
        hidden_states, res, w = forward_crossdown(layer, hidden_states, time, text_embeddings)
        weights += w
        del w
      else:
        hidden_states, res = forward_down(layer, hidden_states, time, text_embeddings)
      samples += res

    for i in range(len(mid_layers)):
      layer = mid_layers[i]
      # print('layer', type(layer))
      # print(type(layer))
      hidden_states, w = forward_crossmid(layer, hidden_states, time, text_embeddings)
      weights += w
      del w

    down_block_samples = samples
    for i in range(len(up_layers)):
      res_samples = down_block_samples[-3:]
      down_block_samples = down_block_samples[: -3]
      layer = up_layers[i]
      # print('layer', type(layer))
      if isinstance(layer, UpBlock2D):
        hidden_states = forward_up(layer, hidden_states, time, text_embeddings, res_samples)
      else:
        hidden_states, w = forward_crossup(layer, hidden_states, time, text_embeddings, res_samples)
        weights += w
        del w

    hidden_states = conv_norm_out(hidden_states)
    hidden_states = conv_act(hidden_states)
    hidden_states = conv_out(hidden_states)
    return hidden_states, samples, time, weights


