import uuid
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.models.attention import BasicTransformerBlock
import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn
import torch.nn.functional as F



epsilon = 0.01
#epsilon = 1e-4



def lrp_conv_old(layer, a, prev, epsilon=epsilon):
  a = (a.data).requires_grad_(True)
  z = layer(a)
  stabilizer = epsilon * torch.sign(z) + (z == 0).float()
  s = (prev/(z+stabilizer)).data
  (z*s.data).sum().backward(retain_graph=True); c = a.grad
  return (a*c).data.detach()


#def lrp_linear(layer, a, prev, epsilon=epsilon, rho=(lambda w, b: (w,b)), incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
#  weight, bias = rho(layer.weight, layer.bias)
#  z = incr(F.linear(a, weight, bias))
#
#  s = prev / z
#  relevance = F.linear(s, weight.t(), bias=None)
#
#  return relevance * a

def lrp_conv(layer, a, R, e=1e-5, rho=lambda w, b: (w, b), incr=lambda w,b:(w,b)):
    """
    General LRP function that handles different layer types.

    Parameters:
        layer: The layer for which LRP is computed.
        a: Input activations to the layer.
        R: Relevance scores at the output of the layer.
        epsilon: Stabilizer term to avoid pivision by zero.
        rho: Function to modify weights according to LRP rules.

    Returns:
        Relevance scores at the input of the layer.
    """
    if R.dtype != layer.weight.dtype:
        R = R.to(layer.weight.dtype)
        a = a.to(layer.weight.dtype)
    if isinstance(layer, torch.nn.Conv2d):
        return lrp_conv2d(layer, a, R, epsilon, rho)
    elif isinstance(layer, torch.nn.Linear):
        return lrp_linear(layer, a, R, epsilon, rho)
    elif isinstance(layer, (torch.nn.ReLU, torch.nn.SiLU, torch.nn.LeakyReLU, torch.nn.GELU)):
        return lrp_activation(layer, a, R)
    elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
        return lrp_normalization(layer, a, R)
    elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
        return lrp_pooling(layer, a, R)
    elif isinstance(layer, torch.nn.Identity):
        # For identity layers, relevance passes through unchanged
        return R
    else:
        # For other layers, use a general method
        return lrp_general(layer, a, R, epsilon)
def lrp_conv2d(layer, a, R, epsilon=epsilon, rho=lambda w, b: (w, b)):
    W, b = rho(layer.weight, layer.bias)

    # forward pass
    z = F.conv2d(a, W, b, layer.stride, layer.padding)
    z_stab = z + epsilon * torch.where(z >= 0, 1., -1.)

    s = R / z_stab                     # proportional share
    c = F.conv_transpose2d(s, W, stride=layer.stride, padding=layer.padding)

    return a * c                       # element-wise relevance

def lrp_linear(layer, a, R, epsilon=epsilon, rho=lambda w, b: (w, b)):
    W, b = rho(layer.weight, layer.bias)

    z = F.linear(a, W, b)
    z_stab = z + epsilon * torch.where(z >= 0, 1., -1.)

    s = R / z_stab
    c = F.linear(s, W.t())             # back-project
    return a * c
def lrp_activation(layer, a, R):
    """
    LRP for activation functions (e.g., ReLU, SiLU, LeakyReLU).

    Parameters:
        layer: The activation layer.
        a: Input activations.
        R: Output relevance scores.

    Returns:
        Relevance scores at the input of the activation function.
    """
    # Compute the gradient of the activation function
    if isinstance(layer, torch.nn.ReLU):
        grad = (a > 0).float()
    elif isinstance(layer, torch.nn.SiLU):
        sigmoid = torch.sigmoid(a)
        grad = sigmoid + a * sigmoid * (1 - sigmoid)
    elif isinstance(layer, torch.nn.LeakyReLU):
        negative_slope = layer.negative_slope
        grad = torch.ones_like(a)
        grad[a < 0] = negative_slope
    elif isinstance(layer, torch.nn.GELU):
        # Approximate derivative of GELU
        grad = torch.sigmoid(1.702 * a)
    else:
        # For other activations, use autograd
        a = a.detach().requires_grad_(True)
        activated = layer(a)
        activated.sum().backward()
        grad = a.grad
        a.grad.zero_()

    # Relevance at the input
    R_input = R * grad

    return R_input
def lrp_normalization(layer, a, R):
    """
    LRP for normalization layers (BatchNorm, GroupNorm, LayerNorm).

    Parameters:
        layer: The normalization layer.
        a: Input activations.
        R: Output relevance scores.

    Returns:
        Relevance scores at the input of the normalization layer.
    """
    # Forward pass
    if isinstance(layer, torch.nn.BatchNorm2d):
        mean = layer.running_mean.view(1, -1, 1, 1)
        var = layer.running_var.view(1, -1, 1, 1)
        gamma = layer.weight.view(1, -1, 1, 1)
        beta = layer.bias.view(1, -1, 1, 1)
        eps = layer.eps

        x_hat = (a - mean) / torch.sqrt(var + eps)
        y = gamma * x_hat + beta

        # Relevance propagation
        R_input = R * (gamma / torch.sqrt(var + eps))

    elif isinstance(layer, torch.nn.LayerNorm):
      a = a.requires_grad_()
      y = layer(a)
      z = y + epsilon * torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))

      relevance_norm = R / z

      grads = torch.autograd.grad(y, a, relevance_norm)[0]
      R_input = grads*a
    else:
        # For other normalization layers, approximate
        R_input = R

    return R_input
def lrp_pooling(layer, a, R):
    """
    LRP for pooling layers (MaxPool2d, AvgPool2d).

    Parameters:
        layer: The pooling layer.
        a: Input activations.
        R: Output relevance scores.

    Returns:
        Relevance scores at the input of the pooling layer.
    """
    if isinstance(layer, torch.nn.MaxPool2d):
        # MaxPool2d
        # Perform max pooling with indices
        with torch.no_grad():
            output, indices = F.max_pool2d(a, kernel_size=layer.kernel_size, stride=layer.stride,
                                           padding=layer.padding, dilation=layer.dilation, return_indices=True)
        # Initialize relevance at input
        R_input = torch.zeros_like(a)
        # Flatten indices and relevance
        indices = indices.view(indices.size(0), -1)
        R_flat = R.view(R.size(0), -1)
        # Scatter relevance back to input positions
        R_input_flat = R_input.view(R_input.size(0), -1)
        R_input_flat.scatter_add_(1, indices, R_flat)
        R_input = R_input_flat.view_as(a)
    elif isinstance(layer, torch.nn.AvgPool2d):
        # AvgPool2d
        # Distribute relevance equally among inputs
        kernel_size = layer.kernel_size
        scale = 1.0 / (kernel_size * kernel_size)
        R_input = F.interpolate(R, scale_factor=layer.kernel_size, mode='nearest') * scale
    else:
        # For other pooling layers
        R_input = R

    return R_input
def lrp_general(layer, a, R, epsilon=1e-5):
    """
    General LRP function using autograd for layers not explicitly handled.

    Parameters:
        layer: The layer.
        a: Input activations.
        R: Output relevance scores.
        epsilon: Stabilizer term.

    Returns:
        Relevance scores at the input of the layer.
    """
    # Use autograd to compute the gradient
    a = a.detach().requires_grad_(True)
    z = layer(a)
    # Stabilize
    z_stable = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    # Compute the proportional relevance
    s = (R / z_stable).data
    # Backward pass
    z.backward(s)
    # Gradient w.r.t. input
    c = a.grad
    # Relevance at the input
    R_input = a * c

    return R_input
def lrp_subtraction(a, b, R_out, epsilon=1e-6):
    """
    LRP for the subtraction operation z = a - b.

    Parameters:
        a: Minuend tensor (x_t).
        b: Subtrahend tensor (predicted_noise).
        R_out: Relevance at the output (denoised_image).
        epsilon: Stabilizer term for numerical stability.

    Returns:
        R_a: Relevance scores for a.
        R_b: Relevance scores for b.
    """
    z = a - b
    stabilizer = epsilon * torch.sign(z) + (z == 0).float()
    s = R_out / (z + stabilizer)
    R_a = s * a
    R_b = -s * b
    return R_a, R_b


#def lrp_conv(layer, a, prev, epsilon=epsilon, rho=(lambda w, b: (w,b)), incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
#  if isinstance(layer, torch.nn.Linear):
#    return lrp_linear(layer, a, prev, epsilon, rho=rho, incr=incr)
#  elif not isinstance(layer, torch.nn.Conv2d):
#    return lrp_conv_old(layer, a, prev, epsilon)
#  weight, bias = rho(layer.weight, layer.bias)
#
#
#  z = incr(F.conv2d(a, weight, bias, layer.stride, layer.padding))
#  s = prev / z
#
#  relevance = F.conv_transpose2d(
#    s,
#    weight,
#    stride=layer.stride,
#    padding=layer.padding
#  )
#  return relevance * a

def reg_rho(weight, bias):
  return weight, bias

def gamma_rho(W, b, gamma=0.2):
    W_pos = torch.clamp(W, min=0)
    return W + gamma * W_pos, b + gamma * torch.clamp(b, min=0) if b is not None else None

#def gamma_rho(weight, bias, gamma=0.8):
#    """
#    Modify weights according to the gamma LRP rule.
#
#    Parameters:
#        weight: Weight tensor of the layer.
#        bias: Bias tensor of the layer (can be None).
#        gamma: Gamma parameter for the gamma LRP rule (gamma >= 0).
#
#    Returns:
#        Modified weight and bias tensors.
#    """
#    # Ensure gamma is non-negative
#    assert gamma >= 0, "Gamma must be non-negative"
#
#    # Compute positive part of weights
#    w_pos = torch.clamp(weight, min=0)
#
#    # Compute modified weights
#    weight_modified = weight + gamma * w_pos
#
#    # Handle bias if not None
#    if bias is not None:
#        # Compute positive part of bias
#        b_pos = torch.clamp(bias, min=0)
#        bias_modified = bias + gamma * b_pos
#    else:
#        bias_modified = None
#
#    return weight_modified, bias_modified
#





def lrp_up(layer, a, prev, samples, time, text_embeddings, tog, block, rho=reg_rho):
  down_samples = samples[0:3]
  resnets = layer.resnets
  upsamplers = layer.upsamplers
  r = []

  gamma=0.1

  if upsamplers is not None:
    for i in range(len(upsamplers)):
      (name, a), layer = next(tog)
      upsampler = upsamplers[i]

      prev = F.interpolate(prev, scale_factor=0.5)
      prev = lrp_conv(upsampler.conv, a, prev, rho=rho)
      r.append(prev)

  for index in reversed(range(len(resnets))):
    # resnet
    rel, time = lrp_resnetmid(resnets[index], a, prev, time, text_embeddings, tog, time, rho)
    r += rel
    prev = r[-1]

    original_a_size = prev.shape[1] - down_samples[-(index + 1)].shape[1]
    prev = prev[:, :original_a_size]
    r.append(prev)


  return r


def lrp_crossup(layer, a, prev, samples, time, text_embeddings, tog, block, res_weights, rho=reg_rho):
  down_samples = samples[0:3]
  resnets = layer.resnets
  upsamplers = layer.upsamplers
  attentions = layer.attentions
  queries = []
  keys = []
  values = []
  weights = []
  r = []





  # (name, a), l = next(tog)
  # while name != f'up_blocks.{block}.resnets.0.norm1':
  #   (name, a), l = next(tog)
  # (name, a), l = next(tog)
  # print(name)
  # a = (a.data).requires_grad_(True)
  # z= forward_crossup(layer, a, time, text_embeddings, down_samples)
  # s = (prev/(z+1e-6)).data
  # (z*s).sum().backward(retain_graph=True); c = a.grad
  # print(c.shape)
  # prev = (a*c).data
  # r.append(prev)
  # return r







  gamma=0.1

  if upsamplers is not None:
    for i in range(len(upsamplers)):
      (name, a), layer = next(tog)
      upsampler = upsamplers[i]

      prev = F.interpolate(prev, scale_factor=0.5)
      prev = lrp_conv(upsampler.conv, a, prev)
      # debug_relevance_scores(prev)
      r.append(prev)

  for index in reversed(range(len(resnets))):
    # transformer
    transformer = attentions[index]
    rel, q, k, v, w = lrp_transformer(transformer, prev, time, text_embeddings, tog, res_weights, rho=rho)
    queries.append(q)
    keys.append(k)
    values.append(v)
    weights.append(w)

    r += rel
    prev = r[-1]
    # debug_relevance_scores(prev)

    # resnet
    rel, time = lrp_resnetmid(resnets[index], a, prev, time, text_embeddings, tog, time, rho=rho)
    r += rel
    prev = r[-1]

    original_a_size = prev.shape[1] - down_samples[-(index + 1)].shape[1]
    prev = prev[:, :original_a_size]
    r.append(prev)
    # debug_relevance_scores(prev)


  return r, queries, keys, values, weights



def lrp_mid(mid, a, prev, time, text_embeddings, tog, prev_time, res_weights, rho=reg_rho):
  r = []
  res1, res2 = mid.resnets
  transformer = mid.attentions[0]

  rel, prev_time = lrp_resnetmid(mid.resnets[1], a, prev, time, text_embeddings, tog, prev_time, rho=rho)
  r+=rel
  prev=r[-1]

  # debug_relevance_scores(prev)
  rel, queries, keys, values, weights = lrp_transformer(mid.attentions[0], prev, time, text_embeddings, tog, res_weights, rho=rho)
  r += rel
  prev = r[-1]
  # debug_relevance_scores(prev)

  rel, prev_time = lrp_resnetmid(mid.resnets[0], a, prev, time, text_embeddings, tog, prev_time, rho=rho)
  r+=rel
  prev=r[-1]
  # debug_relevance_scores(prev)
  return r, prev_time, queries, keys, values, weights



def lrp_down(layer, a, prev, time, text_embeddings, tog, prev_time, rho=reg_rho):
  r = []
  for i in reversed(range(len(layer.resnets))):
    resnet = layer.resnets[i]
    rel, prev_time = lrp_resnetmid(resnet, a, prev, time, text_embeddings, tog, prev_time, rho=rho)
    r+=rel
    prev=r[-1]
    # debug_relevance_scores(prev)
    r += prev
  return r, prev_time



def lrp_crossdown(layer, a, prev, time, text_embeddings, tog, block, prev_time, res_weights, rho=reg_rho):
  r = []
  queries = []
  keys = []
  values = []
  weights = []






  for b in reversed(layer.downsamplers):
    (name, a), l = next(tog)

    prev = lrp_conv_old(b.conv, a, prev)
    # debug_relevance_scores(prev)
    r.append(prev)


  for i in reversed(range(len(layer.resnets))):
    transformer = layer.attentions[i]
    resnet = layer.resnets[i]


    rel, q, k, v, w = lrp_transformer(transformer, prev, time, text_embeddings, tog, res_weights, rho=rho)
    queries.append(q)
    keys.append(k)
    values.append(v)
    weights.append(v)
    r += rel
    prev = r[-1]
    # debug_relevance_scores(prev)



    rel, prev_time = lrp_resnetmid(resnet, a, prev, time, text_embeddings, tog, prev_time, rho=rho)
    r+=rel
    prev=r[-1]
    # debug_relevance_scores(prev)
  # print(len(queries))
  return r, prev_time, queries, keys, values, weights






# def forward_resnetblock(blocklist:ResnetBlock2D, input_tensor, temb):
#   resnet_block = blocklist
#   hidden_states = input_tensor

#   hidden_states = resnet_block.norm1(hidden_states)
#   hidden_states = resnet_block.nonlinearity(hidden_states)

#   hidden_states = resnet_block.conv1(hidden_states)

#   temb = resnet_block.nonlinearity(temb)
#   temb = resnet_block.time_emb_proj(temb)[:, :, None, None]

#   hidden_states = hidden_states + temb

#   hidden_states = resnet_block.norm2(hidden_states)
#   hidden_states = resnet_block.nonlinearity(hidden_states)

#   hidden_states = resnet_block.dropout(hidden_states)
#   hidden_states = resnet_block.conv2(hidden_states)


#   if resnet_block.conv_shortcut is not None:
#       input_tensor = resnet_block.conv_shortcut(input_tensor)

#   output_tensor = (input_tensor + hidden_states) / resnet_block.output_scale_factor
#   return output_tensor

def lrp_resnetmid(resnet_block, a, prev, time, text_embeddings, tog, temb_prev, rho=reg_rho):
  r = []
  tembs = []


  if resnet_block.conv_shortcut is not None:
    (name2, conv2_a), conv2 = next(tog)
  (name3, dropout_a), dropout = next(tog)

  (name4, nonlin1_a), nonlin1 = next(tog)
  (name5, norm2_a), norm2 = next(tog)

  (name6, time_emb_proj_a), time_emb_proj = next(tog)
  (name7, nonlin2_a), nonlin2 = next(tog)

  (name8, conv1_a), conv1 = next(tog)
  (name9, nonlin3_a), nonlin3 = next(tog)
  (name10, norm1_a), norm1 = next(tog)
  (name11, input_tensor_a), input_tensor = next(tog)

  if resnet_block.conv_shortcut is not None:
    # print(norm1_a.shape)
    input_activation = resnet_block.conv_shortcut(norm1_a)
  else:
    input_activation = norm1_a


  # Reverse output scaling
  conv2 = resnet_block.conv2(dropout_a)
  prev, scale_prev = lrp_division((input_activation + conv2), resnet_block.output_scale_factor, prev)
  prev, prev_input = lrp_addition(conv2, input_activation, prev)

  # conv2
  prev = lrp_conv(resnet_block.conv2, dropout_a, prev, rho=rho)
  r.append(prev)


  prev = lrp_conv(nonlin1, norm2_a, prev, rho=rho)
  r.append(prev)


  prev = lrp_conv(resnet_block.norm2, conv1_a, prev, rho=rho)

  temb = resnet_block.nonlinearity(time)
  temb = resnet_block.time_emb_proj(temb)[:, :, None, None]
  prev, prev_temb = lrp_addition(conv1_a, temb, prev)

  prev = lrp_conv(resnet_block.conv1, nonlin3_a, prev, rho=rho)
  r.append(prev)

  # nonlin
  prev = lrp_conv(resnet_block.nonlinearity, norm1_a, prev, rho=rho)
  r.append(prev)

  # norm1
  prev = lrp_conv(resnet_block.norm1, norm1_a, prev, rho=rho)
  r.append(prev)

  return r, time



#   hidden_states = resnet_block.norm1(hidden_states)
#   hidden_states = resnet_block.nonlinearity(hidden_states)

#   hidden_states = resnet_block.conv1(hidden_states)

#   temb = resnet_block.nonlinearity(temb)
#   temb = resnet_block.time_emb_proj(temb)[:, :, None, None]

#   hidden_states = hidden_states + temb

#   hidden_states = resnet_block.norm2(hidden_states)
#   hidden_states = resnet_block.nonlinearity(hidden_states)

#   hidden_states = resnet_block.dropout(hidden_states)
#   hidden_states = resnet_block.conv2(hidden_states)




# def forward_transformer(transformer2D:Transformer2DModel, hidden_states, time, text_embeddings):
#   norm =  transformer2D.norm
#   proj_in = transformer2D.proj_in

#   # in transformer
#   transformer = transformer2D.transformer_blocks[0]
#   proj_out = transformer2D.proj_out
#   residual = hidden_states

#   hidden_states = norm(hidden_states)
#   hidden_states = proj_in(hidden_states)

#   batch, _, height, width = hidden_states.shape
#   inner_dim = hidden_states.shape[1]
#   hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

#   hidden_states = forward_attention(transformer, hidden_states, text_embeddings)

#   hidden_states = (
#       hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
#   )
#   hidden_states = proj_out(hidden_states)
#   return hidden_states + residual

def lrp_transformer(transformer_model, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
  r = []

  proj_out = transformer_model.proj_out
  transformer = transformer_model.transformer_blocks[0]
  proj_in = transformer_model.proj_in
  norm = transformer_model.norm

  (name1, proj_in_a), layer1 = next(tog)
  # print(layer1)
  # print(name1, proj_in_a.shape, layer1, "LAYER1")

  with torch.no_grad():
    attn_layers = [
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
      next(tog),
    ]

  batch, _, height, width = prev.shape
  inner_dim = prev.shape[1]

  (name, norm_a), layer = next(tog)
  (name, conv_shortcut_a), layer = next(tog)
  # print(name, norm_a.shape, layer, 'aneritns')

  proj_in_a = (
      proj_in_a.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
  )
  output = proj_out(proj_in_a)
  prev, prev_norm = lrp_addition(output, conv_shortcut_a, prev)

  # prev = prev - conv_shortcut_a

  # proj_out



  prev = lrp_conv(proj_out, proj_in_a, prev, rho=rho)
  r.append(prev)

  # prev = prev.reshape(batch, height * width, inner_dim).permute(0, 2, 1)

  # prev = prev.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
  prev = prev.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)



  # attention

  rel, queries, keys, values, weights = lrp_attention(transformer_model.transformer_blocks[0], prev, time, text_embeddings, iter(attn_layers), res_weights, rho=rho)
  r += rel
  prev = rel[-1]
  # print(next(tog))

  prev = prev.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

  # proj_in

  prev = lrp_conv(proj_in, norm_a, prev, rho=rho)
  r.append(prev)


  prev = lrp_conv(norm, conv_shortcut_a, prev, rho=rho)
  r.append(prev)
  del attn_layers
  return r, queries, keys, values, weights




# def forward_attention(attn:BasicTransformerBlock, hidden_states, text_embeddings):
#   norm1 = attn.norm1
#   norm2 = attn.norm2
#   norm3 = attn.norm3
#   attn1 = attn.attn1
#   attn2 = attn.attn2
#   ff = attn.ff

#   norm_hidden_states = norm1(hidden_states)
#   attn_output = forward_selfattention(attn1, norm_hidden_states)

#   hidden_states = attn_output + hidden_states

#   norm_hidden_states = norm2(hidden_states)
#   attn_output = forward_crossattention(attn2, norm_hidden_states, text_embeddings)

#   hidden_states = attn_output + hidden_states

#   norm_hidden_states = norm3(hidden_states)
#   ff_output = ff(norm_hidden_states, encoder_hidden_states=text_embeddings)
#   hidden_states =  hidden_states + ff_output

#   return hidden_states


def lrp_attention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
  r = []
  queries = []
  keys = []
  values = []
  weights = []

  (name1, ff_a), layer1 = next(tog)
  (name2, net2_a), layer = next(tog)
  (name3, net1_a), layer = next(tog)
  (name4, proj_a), layer = next(tog)
  (name5, norm3_a), layer4 = next(tog)
  (name6, to_out2_a), layer = next(tog)

  # (name7, to_out1_a), layer = next(tog)
  # (name8, to_v_a), layer = next(tog)
  # (name9, to_k_a), layer = next(tog)
  # (name10, to_q_a), layer = next(tog)
  # (name11, norm2_a), layer11 = next(tog)

  attn2_layers = [
    next(tog),
    next(tog),
    next(tog),
    next(tog),
    next(tog)
  ]

  (name12, to_out1_a), layer = next(tog)
  # print(to_out1_a.shape, 'shape')
  # (name13, to_out2_a), layer = next(tog)
  # (name14, to_v_a), layer = next(tog)
  # (name15, to_k_a), layer = next(tog)
  # (name16, to_q_a), layer = next(tog)

  attn1_layers = [
    next(tog),
    next(tog),
    next(tog),
    next(tog),
    next(tog)
  ]

  # (name17, norm1_a), layer = next(tog)
  (name18, input_a), layer = next(tog)
  # print(name1, ff_a.shape, layer1)
  # print(name5, norm3_a.shape, layer4)
  # print(name11, norm2_a.shape, layer11)
  # print(name18, input_a.shape, layer)
  # print('\n\n\n')
  # print(name1)
  # print(name2)
  # print(name3)
  # print(name4)
  # print(name5)
  # print(name6)
  # print(name7)
  # print(name8)
  # print(name9)
  # print(name10)
  # print(name11)
  # print(name12)
  # print(name13)
  # print(name14)
  # print(name15)
  # print(name16)
  # print(name17)
  # print(name18)
  # print('\n\n\n')

  # print('\n\n\n\n\n')

  # output=prev
  # Reverse feed-forward layer
  # a = (norm3_a.data).requires_grad_(True)
  # z = attn.ff(a, encoder_hidden_states=text_embeddings)
  # # stabilizer = epsilon * torch.sign(z) + (z == 0).float()
  # # s = (prev / (z + stabilizer)).data
  # s = (prev / (z + epsilon)).data
  # (z * s).sum().backward(retain_graph=True)
  # c = a.grad
  # prev = (a * c).data
  prev = lrp_conv(attn.ff, norm3_a, prev, rho=rho)
  r.append(prev)

  # debug_relevance_scores(prev)
  # print('\n')


  prev = lrp_conv(attn.norm3, to_out2_a, prev, epsilon, rho=rho)
  r.append(prev)

  # debug_relevance_scores(prev)
  # print('\n')




  # Reverse addition (attn_output + hidden_states)

  # hidden_states = norm2_a
  # relevance_attn_output = prev * (attn_output / (attn_output + hidden_states + epsilon))
  # relevance_hidden_states = prev * (hidden_states / (attn_output + hidden_states + epsilon))
  # prev = relevance_attn_output + relevance_hidden_states
  # r.append(prev)



  # Reverse cross-attention

  # Reverse self-attention
  # a = (norm2_a.data).requires_grad_(True)
  # z = sd.forward_crossattention(attn.attn2, a, text_embeddings)
  # stabilizer = 1e1 * torch.sign(z) + (z == 0).float()
  # s = (prev / (z+stabilizer)).data
  # (z * s).sum().backward(retain_graph=True)
  # c = a.grad
  # prev = (a * c).data
  # r.append(prev)

  prev_q1, prev_k1, prev_v1, prev_weights1 = lrp_crossattention(attn.attn2, prev, time, text_embeddings, iter(attn2_layers), res_weights, rho=rho)
  prev = prev_q1
  r.append(prev)






  # debug_relevance_scores(prev)
  # print('\n')



  prev = lrp_conv(attn.norm2, to_out1_a, prev, epsilon, rho=rho)
  r.append(prev)

  # debug_relevance_scores(prev)
  # print('\n')


  # Reverse self-attention
  # a = (norm1_a.data).requires_grad_(True)
  # z = sd.forward_selfattention(attn.attn1, a)
  # stabilizer = 1e1 * torch.sign(z) + (z == 0).float()
  # s = (prev / (z+stabilizer)).data
  # (z * s).sum().backward(retain_graph=True)
  # c = a.grad
  # prev = (a * c).data
  # r.append(prev)

  prev_q2, prev_k2, prev_v2, prev_weights2 = lrp_selfattention(attn.attn1, prev, time, text_embeddings, iter(attn1_layers), res_weights, rho=rho)
  prev = prev_q2
  r.append(prev)

  # debug_relevance_scores(prev)
  # print('\n')


  batch, _, height, width = input_a.shape
  inner_dim = input_a.shape[1]
  input_a = input_a.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
  # prev = prev.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

  # print(input_a.shape)
  prev = lrp_conv(attn.norm1, input_a, prev, epsilon, rho=rho)
  r.append(prev)

  # debug_relevance_scores(prev)
  # print('\n\n\n\n\n')

  queries.append((prev_q1, prev_q2))
  keys.append((prev_k1, prev_k2))
  values.append((prev_v1, prev_v2))
  weights.append((prev_weights1, prev_weights2))

  return r, queries, keys, values, weights


def lrp_selfattention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
  (name7, to_out1_a), layer1 = next(tog)
  (name8, to_v_a), layer2 = next(tog)
  (name9, to_k_a), layer3 = next(tog)
  (name10, to_q_a), layer4 = next(tog)
  (name11, norm2_a), layer5 = next(tog)

  inner_dim = to_k_a.shape[-1]
  head_dim = inner_dim // attn.heads
  batch_size, sequence_length, _ = norm2_a.shape

  output = attn.to_out[1](to_out1_a)
  prev, rescale_relevance = lrp_division(output, attn.rescale_output_factor, prev)
  prev = lrp_conv(attn.to_out[1], to_out1_a, prev, rho=rho)

  query = to_q_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
  key = to_k_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
  value = to_v_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

  scale_factor = 1 / math.sqrt(query.size(-1))
  attn_weight1 = torch.matmul(query, key.transpose(-2, -1) * scale_factor)
  attn_weight2 = torch.softmax(attn_weight1, dim=-1)
  output = torch.matmul(attn_weight2, value).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

  prev = lrp_conv(attn.to_out[0], output, prev, rho=rho)
  prev = prev.reshape(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)

  R_query, R_key, R_value, R_weights = lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, (attn_weight1, attn_weight2))

  R_query = R_query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
  R_key = R_key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
  R_value = R_value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

  prev_query = lrp_conv(attn.to_q, norm2_a, R_query, rho=rho)
  prev_key = lrp_conv(attn.to_k, norm2_a, R_key, rho=rho)
  prev_value = lrp_conv(attn.to_v, norm2_a, R_value, rho=rho)

  return prev_query, prev_key.detach().cpu(), prev_value.detach().cpu(), R_weights

def lrp_crossattention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
  (name7, to_out1_a), layer1 = next(tog)
  (name8, to_v_a), layer2 = next(tog)
  (name9, to_k_a), layer3 = next(tog)
  (name10, to_q_a), layer4 = next(tog)
  (name11, norm2_a), layer5 = next(tog)

  inner_dim = to_k_a.shape[-1]
  head_dim = inner_dim // attn.heads
  batch_size, sequence_length, _ = norm2_a.shape

  output = attn.to_out[1](to_out1_a)
  prev, rescale_relevance = lrp_division(output, attn.rescale_output_factor, prev)
  prev = lrp_conv(attn.to_out[1], to_out1_a, prev, rho=rho)

  query = to_q_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
  key = to_k_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
  value = to_v_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

  scale_factor = 1 / math.sqrt(query.size(-1))
  attn_weight1 = torch.matmul(query, key.transpose(-2, -1) * scale_factor)
  attn_weight2 = torch.softmax(attn_weight1, dim=-1)
  output = torch.matmul(attn_weight2, value).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

  prev = lrp_conv(attn.to_out[0], output, prev, rho=rho)
  prev = prev.reshape(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)

  R_query, R_key, R_value, R_weights = lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, (attn_weight1, attn_weight2))

  R_query = R_query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
  R_key = R_key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
  R_value = R_value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

  prev_query = lrp_conv(attn.to_q, norm2_a, R_query, rho=rho)
  prev_key = lrp_conv(attn.to_k, text_embeddings, R_key, rho=rho)
  prev_value = lrp_conv(attn.to_v, text_embeddings, R_value, rho=rho)

  return prev_query, prev_key.detach().cpu(), prev_value.detach().cpu(), R_weights

import math


#def lrp_division(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
#  # Ensure A and B are tensors
#  if not isinstance(A, torch.Tensor):
#    A = torch.tensor(A, dtype=torch.float32)
#  if not isinstance(B, torch.Tensor):
#    B = torch.tensor(B, dtype=torch.float32)
#
#  # Compute the output of the division
#  norm_B = B + epsilon * torch.where(B >= 0, torch.ones_like(B), -torch.ones_like(B))
#  z = A / (norm_B)
#  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#
#
#  # Compute normalized relevance
#  relevance_norm = prev / (z)
#
#  # Distribute the relevance according to the contributions
#  relevance_a = relevance_norm * A
#  relevance_b = -relevance_norm * B * (A / (B ** 2))
#
#  return relevance_a, relevance_b

def lrp_division(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
  # Ensure A and B are tensors
  if not isinstance(A, torch.Tensor):
    A = torch.tensor(A, dtype=torch.float32)
  if not isinstance(B, torch.Tensor):
    B = torch.tensor(B, dtype=torch.float32)

  # Compute the output of the division
  norm_B = B + epsilon * torch.where(B >= 0, torch.ones_like(B), -torch.ones_like(B))
  z = A / (norm_B)
  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

  term_a = A / norm_B
  term_b = -A * B / (norm_B ** 2)

  r_a = (term_a / z) * prev
  r_b = (term_b / z) * prev
  return r_a, r_b

def lrp_multiplication(a, b, R_out):
  # Compute the total contribution
  z = a * b
  total_contrib = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

  # Compute the individual contributions
  contrib_a = (a * b) / total_contrib
  contrib_b = (a * b) / total_contrib
  # Distribute the relevance proportionally
  R_a = (contrib_a / (contrib_a + contrib_b)) * R_out
  R_b = (contrib_b / (contrib_a + contrib_b)) * R_out
  return R_a, R_b

def lrp_addition(A, B, R, epsilon=epsilon):
    Z = A + B
    Z_stab = Z + epsilon * torch.where(Z >= 0, 1., -1.)

    R_A = (A / Z_stab) * R
    R_B = (B / Z_stab) * R
    return R_A, R_B


def lrp_matmul(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
  outputs = torch.matmul(A, B)
  z = outputs * 2
  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

  relevance_norm = prev / z

  relevance_a = torch.matmul(relevance_norm, B.transpose(-1, -2)).mul_(A)
  relevance_b = torch.matmul(A.transpose(-1, -2), relevance_norm).mul_(B)
  return relevance_a, relevance_b


#def lrp_softmax(z, s, R_s, epsilon=1e-9):
#    """
#    Propagate relevance through a softmax layer using a conservation-corrected rule.
#
#    Given:
#      - z: the pre-softmax activations (logits). (Not used explicitly in this rule,
#           but available if you want to extend the rule.)
#      - s: the softmax outputs computed as s = softmax(z)
#      - R_s: the relevance scores at the softmax output.
#
#    The idea is to redistribute R_s back to the logits in proportion to s.
#    However, a direct redistribution as s * R_s generally does not conserve relevance.
#    Therefore, we add a correction term such that:
#
#         R_z = s * (R_s + correction)
#
#    with correction = (R_s_total - (s * R_s).sum(dim=-1, keepdim=True)),
#    where R_s_total is the total relevance at the softmax output.
#
#    This ensures that:
#         R_z.sum(dim=-1) = R_s_total
#    i.e. relevance is conserved.
#
#    Parameters:
#      z (torch.Tensor): Pre-softmax activations (logits), shape [B, N].
#      s (torch.Tensor): Softmax outputs, shape [B, N].
#      R_s (torch.Tensor): Relevance at the softmax output, shape [B, N].
#      epsilon (float): Small constant for numerical stability (unused in this simple form).
#
#    Returns:
#      torch.Tensor: Relevance scores for the logits, shape [B, N].
#    """
#    # Total relevance at the softmax output for each sample
#    R_s_total = R_s.sum(dim=-1, keepdim=True)
#    # Compute the partial relevance that would be assigned by a naive redistribution
#    naive = s * R_s
#    # The correction needed to ensure conservation:
#    correction = R_s_total - naive.sum(dim=-1, keepdim=True)
#    # Redistribute using softmax probabilities plus correction
#    R_z = s * (R_s + correction)
#    return R_z

#def lrp_softmax(z_logits, s_softmax, R_s):
#    # Ensure relevance mass equals 1 per sample before back-prop
#    R_s = R_s / (R_s.sum(dim=-1, keepdim=True) + 1e-6)
#
#    # Conservative redistribution (Arras et al.)
#    R_z = s_softmax * (R_s - (s_softmax * R_s).sum(-1, keepdim=True))
#    return R_z

def lrp_softmax(inputs, output, output_relevance):
  with torch.no_grad():
    inputs = torch.where(torch.isneginf(inputs), torch.tensor(0).to(inputs), inputs)

    relevance = inputs * (output_relevance - output * output_relevance.sum(-1, keepdim=True))
    return relevance

def lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, attn_weights):
  attn_weight1, attn_weight2 = attn_weights

  scale_factor = 1 / math.sqrt(query.size(-1))

  output_rel = prev
  R_weights, R_value = lrp_matmul(attn_weight2, value, output_rel)

  relevance_attn_weight = lrp_softmax(attn_weight1, attn_weight2, R_weights)

  R_query, R_key = lrp_matmul(query, key.transpose(-2, -1), relevance_attn_weight)
  #del relevance_attn_weight
  del attn_weight1, attn_weight2
  return R_query.detach(), R_key.detach(), R_value.detach(), relevance_attn_weight.detach()




#def norm(relevance):
#  abs_relevance = relevance.abs()
#  min_val = abs_relevance.min()
#  max_val = abs_relevance.max()
#  normalized = (abs_relevance + min_val) / (max_val + min_val)
#  return normalized * torch.sign(relevance)


def norm(relevance):
  max_value = 1e10
  # Replace positive and negative infinities with max_value and -max_value respectively
  tensor = torch.where(torch.isinf(relevance), torch.sign(relevance) * max_value, relevance)
  # Replace NaNs with zero
  tensor = torch.where(torch.isnan(relevance), torch.zeros_like(relevance), relevance)
  return tensor

def norm_rel(relevance):
  # Separate positive and negative relevance scores
  positive_relevance = torch.clamp(relevance, min=0.0)
  negative_relevance = torch.clamp(relevance, max=0.0).abs()

  # Normalize positive relevance scores
  if positive_relevance.max() > 0:
      positive_norm = positive_relevance / positive_relevance.max()
  else:
      positive_norm = positive_relevance

  # Normalize negative relevance scores
  if negative_relevance.max() > 0:
      negative_norm = negative_relevance / negative_relevance.max()
  else:
      negative_norm = negative_relevance

  # Combine normalized scores with appropriate signs
  normalized_relevance = positive_norm - negative_norm

  return normalized_relevance

def neg_heatmap(relevance, image, mult=1):
  lrp_scores = relevance.unsqueeze(dim=-1)
  scores = lrp_scores.sign()
  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
  scores = torch.cat((scores, scores, scores), dim=-1)

  red = norm_rel(lrp_scores)*torch.Tensor([255*mult, 1, 1])
  blue = norm_rel(-lrp_scores)*torch.Tensor([1, 1, 255*mult])


  image1 = torch.where(scores>0, image, blue).to(torch.uint8).cpu()

  plt.figure(figsize=(5,5))
  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
  plt.axis('off')
  plt.imshow(image1.numpy())
  plt.show()


def pos_heatmap(relevance, image, mult=1):
  lrp_scores = relevance.unsqueeze(dim=-1)
  scores = lrp_scores.sign()
  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
  scores = torch.cat((scores, scores, scores), dim=-1)

  red = norm_rel(lrp_scores)*torch.Tensor([255*mult, 1, 1])
  blue = norm_rel(-lrp_scores)*torch.Tensor([1, 1, 255*mult])


  image1 = torch.where(scores>0, red, image).to(torch.uint8).cpu()


  plt.figure(figsize=(5,5))
  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
  plt.axis('off')
  plt.imshow(image1.numpy())
  plt.show()



def comb_heatmap(relevance, mult=1):
  lrp_scores = relevance.unsqueeze(dim=-1)
  scores = lrp_scores.sign()
  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
  scores = torch.cat((scores, scores, scores), dim=-1)

  red = (norm_rel(lrp_scores))*torch.Tensor([255*mult, 1, 1])
  blue = (norm_rel(-lrp_scores))*torch.Tensor([1, 1, 255*mult])

  image1 = torch.where(scores>0, red, blue).to(torch.uint8).cpu()


  plt.figure(figsize=(5,5))
  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
  plt.axis('off')
  plt.imshow(image1.numpy())
  plt.show()



# def norm(relevance):
#   return ((relevance.abs() + relevance.abs().min()) / ((relevance).abs().max() + (relevance).abs().min())) * torch.sign(relevance)


# def normalize(gradients):
#   scale_factor=0.1
#   epsilon=1e1
#   max_grad = gradients.abs().max()
#   return gradients * (scale_factor / (max_grad + epsilon))


# def normalize(tens):
#   return tens / (tens.abs().max() + 1e-6)

def debug_relevance_scores(relevance):
  print("Relevance Min:", relevance.min().item())
  print("Relevance Max:", relevance.max().item())
  print("Relevance Mean:", relevance.mean().item())
  print("Relevance Std Dev:", relevance.std().item())

def visualize_relevance(decoded_relevance, original_image, save_path=None):
    # Normalize the relevance maps
    relevance_map = decoded_relevance.squeeze().cpu().numpy()
    relevance_map = logarithmic_mapping(relevance_map)

    maxi = relevance_map.max()
    mini = relevance_map.min()

    # Overlay the relevance map on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.imshow(relevance_map,vmin=mini, vmax=maxi, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title("Relevance Map Overlay")

    if save_path:
        plt.savefig(save_path)
    plt.show()



import numpy
from matplotlib.colors import LogNorm, ListedColormap, Normalize, TwoSlopeNorm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import cv2

def overlay_attention_on_image(image, attn_weights, alpha=0.6, cmap='jet'):
    """
    Overlay a heatmap derived from attention weights on an image.

    Parameters:
        image (np.array): The original image as an array of shape (H, W, 3) with values [0, 255].
        attn_weights (torch.Tensor or np.array): A 2D attention map with shape (h, w).
        alpha (float): Transparency for the heatmap overlay.
        cmap (str): Name of the matplotlib colormap to use.

    Returns:
        np.array: The image with the attention heatmap overlay.
    """
    # Convert attn_weights to numpy if it's a tensor
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    # Normalize the attention weights to [0, 1]
    attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)

    # Resize the attention map to the image dimensions
    heatmap = cv2.resize(attn_norm, (image.shape[1], image.shape[0]))
    print(heatmap)

    # Apply a colormap to the attention map to create an RGB heatmap
    heatmap_color = plt.get_cmap(cmap)(heatmap)  # returns RGBA
    heatmap_color = np.delete(heatmap_color, 3, axis=2)  # remove the alpha channel

    # Convert the image to float in [0, 1] if it's not already
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image

    # Combine the heatmap with the image using alpha blending
    overlay = np.concatenate(((heatmap_color * alpha), (image_float * (1 - alpha))))
    overlay = np.clip(overlay, 0, 1)

    # Convert back to uint8 for display
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8

#def heatmap(relevance, sx, sy, log=False, save_path='save.png'):
#    if log:
#        normalized_relevance = logarithmic_mapping(relevance)
#    else:
#        normalized_relevance = norm(relevance)
#
#    relevance_np = normalized_relevance.squeeze().cpu().numpy()
#
#    # Define color map
#    cmap = plt.cm.seismic
#    plt.figure(figsize=(sx, sy))
#    plt.imshow(relevance_np, cmap=cmap, interpolation='nearest')
#    plt.axis('off')
#    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#    plt.show()

def heatmap(R,sx,sy, log=False, save_path='save.png'):
  if log:
    R = logarithmic_mapping(R)
    # b = 5*(numpy.abs(R)).mean()
    # R = norm(R)
    maxi = R.max()
    mini = R.min()
  else:
    #mini = -8*((numpy.abs(R)**3.0).mean()**(1.0/3))
    #maxi = 8*((numpy.abs(R)**3.0).mean()**(1.0/3))
    R = norm_rel(R)
    maxi = R.max()
    mini = R.min()


  my_cmap = plt.cm.seismic
  plt.figure(figsize=(sx,sy))
  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
  plt.axis('off')
  plt.imshow(R,cmap=my_cmap,interpolation='nearest')
  plt_id = uuid.uuid4()
  plt.savefig(save_path)

  plt.show()
def logarithmic_mapping(relevance_scores, scale_factor=1):
    """
    Apply a logarithmic transformation to LRP relevance scores to compress extreme values,
    ensuring that large positive or negative scores do not dominate the overall contribution.

    The function maps each score using the formula:
        mapped_score = sign(score) * log(1 + abs(score) / scale_factor)
    This transformation preserves the sign of the original score while compressing its magnitude.

    Parameters:
        relevance_scores (array-like): A list or NumPy array of relevance scores.
        scale_factor (float): A constant used to adjust the scaling sensitivity. Default is 155.0.

    Returns:
        np.ndarray: A NumPy array of the logarithmically mapped relevance scores.
    """
    relevance_scores = np.array(relevance_scores)
    mapped_scores = np.sign(relevance_scores) * np.log1p(np.abs(relevance_scores) / scale_factor)
    return mapped_scores

def logarithmic_mapping_torch(relevance_scores, scale_factor=1):
    """
    Apply a logarithmic transformation to LRP relevance scores using PyTorch,
    compressing extreme values while preserving sign.

    The transformation is:
        mapped_score = sign(score) * log(1 + abs(score) / scale_factor)

    Parameters:
        relevance_scores (tensor or array-like): A tensor or array of relevance scores.
        scale_factor (float): Scaling constant. Default is 1 (adjust as needed).

    Returns:
        torch.Tensor: A tensor of logarithmically mapped relevance scores.
    """
    if not isinstance(relevance_scores, torch.Tensor):
        relevance_scores = torch.tensor(relevance_scores, dtype=torch.float32)

    mapped_scores = torch.sign(relevance_scores) * torch.log1p(torch.abs(relevance_scores) / scale_factor)
    return mapped_scores

#def logarithmic_mapping(relevance_scores, scale_factor=155.0):
#    # Small epsilon to prevent log(0)
#
#    # Separate positive and negative relevance scores
#    positive_relevance = torch.clamp(relevance_scores, min=epsilon)
#    negative_relevance = torch.clamp(relevance_scores, max=epsilon).abs()
#
#    # Apply logarithmic scaling to positive relevance
#    positive_log = torch.log(positive_relevance)
#    if positive_log.max() > 0:
#        positive_norm = positive_log / positive_log.max()
#    else:
#        positive_norm = positive_log
#
#    # Apply logarithmic scaling to negative relevance
#    negative_log = torch.log(negative_relevance)
#    print(negative_log)
#
#    if negative_log.max() > 0:
#        negative_norm = negative_log / negative_log.max()
#    else:
#        negative_norm = negative_log
#
#    # Combine normalized logs with appropriate signs
#    negative_norm = torch.where(negative_norm <0, 0, negative_norm);
#    positive_norm = torch.where(positive_norm <0, 0, positive_norm);
#
#
#    normalized_relevance = positive_norm - negative_norm
#
#
#    # Apply scaling factor if desired
#    #normalized_relevance *= scale_factor
#
#    return normalized_relevance

#def logarithmic_mapping(relevance_scores, scale_factor=3):
#  sign = torch.sign(relevance_scores)
#  adjusted_scores = torch.where(relevance_scores == 0, epsilon, relevance_scores)
#  adjusted_scores =  torch.log(adjusted_scores.abs())
#  adjusted_scores += adjusted_scores.min().abs()
#  adjusted_scores *= sign
#  return adjusted_scores

def save_tensor_to_mmap(tensor, filename):
  np_tensor = tensor.cpu().numpy()
  shape = np_tensor.shape
  dtype = np_tensor.dtype

  # Create a memory-mapped file with write access
  mmap_file = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
  mmap_file[:] = np_tensor[:]
  mmap_file.flush()
  del mmap_file  # Ensure changes are written to disk

# Function to load a tensor from a memory-mapped file
def load_tensor_from_mmap(filename, shape, dtype):
    # Open the memory-mapped file with read access
    mmap_file = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    tensor = torch.tensor(mmap_file)
    del mmap_file
    return tensor

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize_text_relevance(text_tokens, relevance_scores, cmap_name='coolwarm', save_path='save.png'):
    # Normalize relevance scores between 0 and 1
    relevance_scores = relevance_scores.cpu().numpy()
    max_abs_score = np.max(np.abs(relevance_scores))
    normalized_scores = relevance_scores;


    # Create a color map
    cmap = plt.get_cmap(cmap_name)

    # Plot each token with its corresponding color
    fig, ax = plt.subplots(figsize=(len(text_tokens) * 0.5, 1))
    ax.axis('off')

    for i, (token, score) in enumerate(zip(text_tokens, normalized_scores)):
        color = cmap(score)
        ax.text(i, 0, token, fontsize=12, color=color, ha='center', va='center', rotation=45)

    # Adjust plot limits
    ax.set_xlim(-0.5, len(text_tokens) - 0.5)
    ax.set_ylim(-0.5, 0.5)


    file = open(f'{save_path.replace(".png", "")}_scores.txt', 'w+')
    np.savetxt(f'{save_path.replace(".png", "")}_scores.txt', np.array(normalized_scores))
    file.close()

    # Save and display the plot
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

#def visualize_text_relevance(text_tokens, norm_scores, cmap_name='coolwarm', save_path='save.png'):
#  # Normalize the relevance scores to be between 0 and 1
#  # print(norm_scores)
#  sign = np.sign(norm_scores.cpu().numpy())
#  # norm_scores = logarithmic_mapping(norm_scores)
#
#  # norm_scores = norm(norm_scores)
#
#  # Create a color map
#  cmap = plt.get_cmap(cmap_name)
#
#  # Create a figure and axis
#  fig, ax = plt.subplots()
#
#  # Plot each word with its corresponding color
#  new_scrores = []
#  for i, token in enumerate(text_tokens):
#    if token=='<|endoftext|>':
#      break;
#    if token=='<|startoftext|>':
#      continue;
#    new_scrores.append(norm_scores[i])
#    ax.text(i/5, 0.5, token, fontsize=12, color=cmap(norm_scores[i]), ha='center', va='center')
#
#  # Remove the axis
#  ax.axis('off')
#
#  # Show the plot
#  text_id = uuid.uuid4()
#  plt.savefig(save_path)
#  file = open(f'{save_path.replace(".png", "")}_scores.txt', 'w+')
#  np.savetxt(f'{save_path.replace(".png", "")}_scores.txt', np.array(new_scrores))
#  file.close()
#
#  plt.show()
#

def apply_lrp(unet, vae, layers, activations, samples, time, text_embeddings, initial_input, weights):
  unet = unet
  vae = vae
  queries = []

  keys = []
  values = []
  #attn_weights = []

  temp = range(len(layers[5:563]))
  L = len(temp)
  lays = layers[5:563][::-1]
  tog = zip(([None]+activations[5:562])[::-1], lays)

  down_blocks = unet.down_blocks
  mid_block = unet.mid_block
  up_blocks = unet.up_blocks

  conv_norm_out = unet.conv_norm_out
  conv_act = unet.conv_act
  conv_in = unet.conv_in

  (name, a), layer = next(tog)
  prev = activations[562][1]
  basic_incr = lambda z: z+1e-9
  prev = lrp_conv(layer, a, prev, incr=basic_incr)


  (name, a), layer = next(tog)
  prev = lrp_conv(layer, a, prev, incr=basic_incr)


  (name, a), layer = next(tog)
  prev = lrp_conv(layer, a, prev, incr=basic_incr)


  down_block_samples = samples
  res_samples = down_block_samples[0:3]
  down_block_samples = down_block_samples[3:]


  res_weights = weights[-2:]
  weights = weights[:-2]
  r, q, k, v, w = lrp_crossup(up_blocks[-1], a, prev, res_samples, time, text_embeddings, tog, 3, res_weights, reg_rho)

  prev = norm(r[-1])
  del res_weights

  queries += q
  keys += k
  values += v
  #attn_weights += w

  del r,q,k,v, w


  res_samples = down_block_samples[0:3]
  down_block_samples = down_block_samples[3:]

  res_weights = weights[-2:]
  weights = weights[:-2]
  r, q, k, v, w = lrp_crossup(up_blocks[-2], a, prev, res_samples, time, text_embeddings, tog, 2, res_weights, reg_rho)


  queries += q
  keys += k
  values += v
  #attn_weights += w

  del res_weights
  prev = norm(r[-1])
  del r,q,k,v, w

  res_samples = down_block_samples[0:3]
  down_block_samples = down_block_samples[3:]
  res_weights = weights[-2:]
  weights = weights[:-2]
  r, q, k, v, w = lrp_crossup(up_blocks[-3], a, prev, res_samples, time, text_embeddings, tog, 1, res_weights, reg_rho)

  queries += q
  keys += k
  values += v
  #attn_weights += w

  del res_weights
  prev = norm(r[-1])
  del r,q,k,v, w



  res_samples = down_block_samples[0:3]
  down_block_samples = down_block_samples[3:]
  r = lrp_up(up_blocks[0], a, prev, res_samples, time, text_embeddings, tog, 1, reg_rho)
  prev = norm(r[-1])
  del r

  res_weights = weights[-2:]
  weights = weights[:-2]
  r, time_prev, q, k, v, w = lrp_mid(mid_block, a, prev, time, text_embeddings, tog, time, res_weights,  rho=gamma_rho)

  queries.append(q)
  keys.append(k)
  values.append(v)
  #attn_weights.append(w)

  del res_weights
  prev = norm(r[-1])
  del r,q,k,v,w


  r, time_prev = lrp_down(down_blocks[-1], a, prev, time, text_embeddings, tog, time_prev,  rho=gamma_rho)
  prev = norm(r[-1])
  del r


  res_weights = weights[-2:]
  weights = weights[:-2]
  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-2], a, prev, time, text_embeddings, tog, 2, time_prev, res_weights,  rho=gamma_rho)

  queries += q
  keys += k
  values += v
  #attn_weights += w

  del res_weights
  prev = norm(r[-1])
  del r,q,k,v,w

  res_weights = weights[-2:]
  weights = weights[:-2]
  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-3], a, prev, time, text_embeddings, tog, 1, time_prev, res_weights, rho=gamma_rho)


  queries += q
  keys += k
  values += v
  #attn_weights += w

  del res_weights
  prev = norm(r[-1])
  del r,q,k,v


  res_weights = weights[-2:]
  weights = weights[:-2]
  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-4], a, prev, time, text_embeddings, tog, 0, time_prev, res_weights, rho=gamma_rho)
  del res_weights
  prev = norm(r[-1])
  queries += q
  keys += k
  #attn_weights += w

  values += v
  del r,q,k,v, w

  r = lrp_conv(conv_in, initial_input, prev, rho=gamma_rho)
  del tog

  return r, queries, keys, values, [] #attn_weights
#import uuid
#from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
#from diffusers.models.attention import BasicTransformerBlock
#import torch
#import matplotlib.pyplot as plt
#import numpy as np
#from diffusers.models.resnet import ResnetBlock2D
#from diffusers.models.transformers.transformer_2d import Transformer2DModel
#from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn
#import torch.nn.functional as F
#
#
#
#epsilon = 0.01
#
#
#def lrp_conv_old(layer, a, prev, epsilon=epsilon): 
#  a = (a.data).requires_grad_(True)
#  z = layer(a)
#  stabilizer = epsilon * torch.sign(z) + (z == 0).float()
#  s = (prev/(z+stabilizer)).data              
#  (z*s.data).sum().backward(retain_graph=True); c = a.grad       
#  return (a*c).data.detach() 
#
#
##def lrp_linear(layer, a, prev, epsilon=epsilon, rho=(lambda w, b: (w,b)), incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
##  weight, bias = rho(layer.weight, layer.bias)
##  z = incr(F.linear(a, weight, bias))
##
##  s = prev / z
##  relevance = F.linear(s, weight.t(), bias=None)
##
##  return relevance * a
#
#def lrp_conv(layer, a, R, e=1e-5, rho=lambda w, b: (w, b), incr=lambda w,b:(w,b)):
#    """
#    General LRP function that handles different layer types.
#
#    Parameters:
#        layer: The layer for which LRP is computed.
#        a: Input activations to the layer.
#        R: Relevance scores at the output of the layer.
#        epsilon: Stabilizer term to avoid pivision by zero.
#        rho: Function to modify weights according to LRP rules.
#
#    Returns:
#        Relevance scores at the input of the layer.
#    """
#    if isinstance(layer, torch.nn.Conv2d):
#        return lrp_conv2d(layer, a, R, epsilon, rho)
#    elif isinstance(layer, torch.nn.Linear):
#        return lrp_linear(layer, a, R, epsilon, rho)
#    elif isinstance(layer, (torch.nn.ReLU, torch.nn.SiLU, torch.nn.LeakyReLU, torch.nn.GELU)):
#        return lrp_activation(layer, a, R)
#    elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
#        return lrp_normalization(layer, a, R)
#    elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
#        return lrp_pooling(layer, a, R)
#    elif isinstance(layer, torch.nn.Identity):
#        # For identity layers, relevance passes through unchanged
#        return R
#    else:
#        # For other layers, use a general method
#        return lrp_general(layer, a, R, epsilon)
#def lrp_conv2d(layer, a, R, epsilon=1e-5, rho=lambda w, b: (w, b)):
#    """
#    LRP for Conv2D layers.
#
#    Parameters:
#        layer: The Conv2D layer.
#        a: Input activations.
#        R: Output relevance scores.
#        epsilon: Stabilizer term.
#        rho: Weight modification function.
#
#    Returns:
#        Relevance scores at the input of the Conv2D layer.
#    """
#    weight, bias = rho(layer.weight, layer.bias)
#
#    # Forward pass
#    z = F.conv2d(a, weight, bias, stride=layer.stride, padding=layer.padding)
#    # Stabilize to avoid division by zero
#    z_stable = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#    # Compute the relevance scores proportionality
#    s = R / z_stable
#    # Backward pass
#    c = F.conv_transpose2d(s, weight, stride=layer.stride, padding=layer.padding)
#    # Relevance at the input
#    R_input = a * c
#
#    return R_input
#def lrp_linear(layer, a, R, epsilon=1e-5, rho=lambda w, b: (w, b)):
#    weight, bias = rho(layer.weight, layer.bias)
#    z = F.linear(a, weight, bias)
#    z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#    relevance_norm = R / z
#    relevance = torch.matmul(relevance_norm, layer.weight).mul_(a)
#    return relevance
#    
#    
#    weight, bias = rho(layer.weight, layer.bias)
#
#    # Forward pass
#    z = F.linear(a, weight, bias)
#    # Stabilize
#    z_stable = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#    # Compute the relevance scores proportionality
#    s = R / z_stable
#    # Backward pass
#    c = F.linear(s, weight.t())
#    # Relevance at the input
#    R_input = a * c
#
#    return R_input
#def lrp_activation(layer, a, R):
#    """
#    LRP for activation functions (e.g., ReLU, SiLU, LeakyReLU).
#
#    Parameters:
#        layer: The activation layer.
#        a: Input activations.
#        R: Output relevance scores.
#
#    Returns:
#        Relevance scores at the input of the activation function.
#    """
#    # Compute the gradient of the activation function
#    if isinstance(layer, torch.nn.ReLU):
#        grad = (a > 0).float()
#    elif isinstance(layer, torch.nn.SiLU):
#        sigmoid = torch.sigmoid(a)
#        grad = sigmoid + a * sigmoid * (1 - sigmoid)
#    elif isinstance(layer, torch.nn.LeakyReLU):
#        negative_slope = layer.negative_slope
#        grad = torch.ones_like(a)
#        grad[a < 0] = negative_slope
#    elif isinstance(layer, torch.nn.GELU):
#        # Approximate derivative of GELU
#        grad = torch.sigmoid(1.702 * a)
#    else:
#        # For other activations, use autograd
#        a = a.detach().requires_grad_(True)
#        activated = layer(a)
#        activated.sum().backward()
#        grad = a.grad
#        a.grad.zero_()
#
#    # Relevance at the input
#    R_input = R * grad
#
#    return R_input
#def lrp_normalization(layer, a, R):
#    """
#    LRP for normalization layers (BatchNorm, GroupNorm, LayerNorm).
#
#    Parameters:
#        layer: The normalization layer.
#        a: Input activations.
#        R: Output relevance scores.
#
#    Returns:
#        Relevance scores at the input of the normalization layer.
#    """
#    # Forward pass
#    if isinstance(layer, torch.nn.BatchNorm2d):
#        mean = layer.running_mean.view(1, -1, 1, 1)
#        var = layer.running_var.view(1, -1, 1, 1)
#        gamma = layer.weight.view(1, -1, 1, 1)
#        beta = layer.bias.view(1, -1, 1, 1)
#        eps = layer.eps
#
#        x_hat = (a - mean) / torch.sqrt(var + eps)
#        y = gamma * x_hat + beta
#
#        # Relevance propagation
#        R_input = R * (gamma / torch.sqrt(var + eps))
#
#    elif isinstance(layer, torch.nn.LayerNorm):
#      a = a.requires_grad_()
#      y = layer(a)
#      z = y + epsilon * torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
#
#      relevance_norm = R / z
#
#      grads = torch.autograd.grad(y, a, relevance_norm)[0]
#      R_input = grads*a
#    else:
#        # For other normalization layers, approximate
#        R_input = R
#
#    return R_input
#def lrp_pooling(layer, a, R):
#    """
#    LRP for pooling layers (MaxPool2d, AvgPool2d).
#
#    Parameters:
#        layer: The pooling layer.
#        a: Input activations.
#        R: Output relevance scores.
#
#    Returns:
#        Relevance scores at the input of the pooling layer.
#    """
#    if isinstance(layer, torch.nn.MaxPool2d):
#        # MaxPool2d
#        # Perform max pooling with indices
#        with torch.no_grad():
#            output, indices = F.max_pool2d(a, kernel_size=layer.kernel_size, stride=layer.stride,
#                                           padding=layer.padding, dilation=layer.dilation, return_indices=True)
#        # Initialize relevance at input
#        R_input = torch.zeros_like(a)
#        # Flatten indices and relevance
#        indices = indices.view(indices.size(0), -1)
#        R_flat = R.view(R.size(0), -1)
#        # Scatter relevance back to input positions
#        R_input_flat = R_input.view(R_input.size(0), -1)
#        R_input_flat.scatter_add_(1, indices, R_flat)
#        R_input = R_input_flat.view_as(a)
#    elif isinstance(layer, torch.nn.AvgPool2d):
#        # AvgPool2d
#        # Distribute relevance equally among inputs
#        kernel_size = layer.kernel_size
#        scale = 1.0 / (kernel_size * kernel_size)
#        R_input = F.interpolate(R, scale_factor=layer.kernel_size, mode='nearest') * scale
#    else:
#        # For other pooling layers
#        R_input = R
#
#    return R_input
#def lrp_general(layer, a, R, epsilon=1e-5):
#    """
#    General LRP function using autograd for layers not explicitly handled.
#
#    Parameters:
#        layer: The layer.
#        a: Input activations.
#        R: Output relevance scores.
#        epsilon: Stabilizer term.
#
#    Returns:
#        Relevance scores at the input of the layer.
#    """
#    # Use autograd to compute the gradient
#    a = a.detach().requires_grad_(True)
#    z = layer(a)
#    # Stabilize
#    z_stable = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#    # Compute the proportional relevance
#    s = (R / z_stable).data
#    # Backward pass
#    z.backward(s)
#    # Gradient w.r.t. input
#    c = a.grad
#    # Relevance at the input
#    R_input = a * c
#
#    return R_input
#def lrp_subtraction(a, b, R_out, epsilon=1e-6):
#    """
#    LRP for the subtraction operation z = a - b.
#    
#    Parameters:
#        a: Minuend tensor (x_t).
#        b: Subtrahend tensor (predicted_noise).
#        R_out: Relevance at the output (denoised_image).
#        epsilon: Stabilizer term for numerical stability.
#        
#    Returns:
#        R_a: Relevance scores for a.
#        R_b: Relevance scores for b.
#    """
#    z = a - b
#    stabilizer = epsilon * torch.sign(z) + (z == 0).float()
#    s = R_out / (z + stabilizer)
#    R_a = s * a
#    R_b = -s * b
#    return R_a, R_b
#
#
##def lrp_conv(layer, a, prev, epsilon=epsilon, rho=(lambda w, b: (w,b)), incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
##  if isinstance(layer, torch.nn.Linear):
##    return lrp_linear(layer, a, prev, epsilon, rho=rho, incr=incr)
##  elif not isinstance(layer, torch.nn.Conv2d):
##    return lrp_conv_old(layer, a, prev, epsilon)
##  weight, bias = rho(layer.weight, layer.bias)
##
##
##  z = incr(F.conv2d(a, weight, bias, layer.stride, layer.padding))
##  s = prev / z
##
##  relevance = F.conv_transpose2d(
##    s,
##    weight,
##    stride=layer.stride,
##    padding=layer.padding
##  )
##  return relevance * a
#
#def reg_rho(weight, bias):
#  return weight, bias
#
#def gamma_rho(weight, bias, gamma=.2):
#  weight = weight + weight * torch.max(torch.tensor(0.,device=weight.device), weight) * gamma
#  if bias is not None:
#    bias = bias + bias * torch.max(torch.tensor(0.,device=bias.device), bias) * gamma
#  return weight, bias
##def gamma_rho(weight, bias, gamma=0.8):
##    """
##    Modify weights according to the gamma LRP rule.
##
##    Parameters:
##        weight: Weight tensor of the layer.
##        bias: Bias tensor of the layer (can be None).
##        gamma: Gamma parameter for the gamma LRP rule (gamma >= 0).
##
##    Returns:
##        Modified weight and bias tensors.
##    """
##    # Ensure gamma is non-negative
##    assert gamma >= 0, "Gamma must be non-negative"
##
##    # Compute positive part of weights
##    w_pos = torch.clamp(weight, min=0)
##
##    # Compute modified weights
##    weight_modified = weight + gamma * w_pos
##
##    # Handle bias if not None
##    if bias is not None:
##        # Compute positive part of bias
##        b_pos = torch.clamp(bias, min=0)
##        bias_modified = bias + gamma * b_pos
##    else:
##        bias_modified = None
##
##    return weight_modified, bias_modified
##
#
#
#
#
#
#def lrp_up(layer, a, prev, samples, time, text_embeddings, tog, block, rho=reg_rho):
#  down_samples = samples[0:3] 
#  resnets = layer.resnets
#  upsamplers = layer.upsamplers 
#  r = []
#  
#  gamma=0.1
#  
#  if upsamplers is not None:
#    for i in range(len(upsamplers)):
#      (name, a), layer = next(tog)
#      upsampler = upsamplers[i] 
#      
#      prev = F.interpolate(prev, scale_factor=0.5)
#      prev = lrp_conv(upsampler.conv, a, prev, rho=rho)
#      r.append(prev)
#
#  for index in reversed(range(len(resnets))):    
#    # resnet
#    rel, time = lrp_resnetmid(resnets[index], a, prev, time, text_embeddings, tog, time, rho)
#    r += rel
#    prev = r[-1]
#    
#    original_a_size = prev.shape[1] - down_samples[-(index + 1)].shape[1]
#    prev = prev[:, :original_a_size] 
#    r.append(prev)
#    
#
#  return r
#  
#  
#def lrp_crossup(layer, a, prev, samples, time, text_embeddings, tog, block, res_weights, rho=reg_rho):
#  down_samples = samples[0:3] 
#  resnets = layer.resnets
#  upsamplers = layer.upsamplers
#  attentions = layer.attentions
#  queries = []
#  keys = []
#  values = []
#  weights = []
#  r = []
#
#
#
#  
#  
#  # (name, a), l = next(tog)
#  # while name != f'up_blocks.{block}.resnets.0.norm1':
#  #   (name, a), l = next(tog)
#  # (name, a), l = next(tog)
#  # print(name)
#  # a = (a.data).requires_grad_(True)
#  # z= forward_crossup(layer, a, time, text_embeddings, down_samples)
#  # s = (prev/(z+1e-6)).data                                    
#  # (z*s).sum().backward(retain_graph=True); c = a.grad    
#  # print(c.shape)     
#  # prev = (a*c).data         
#  # r.append(prev)     
#  # return r
#
#
#
#
#
#
#
#  gamma=0.1
#  
#  if upsamplers is not None:
#    for i in range(len(upsamplers)):
#      (name, a), layer = next(tog)
#      upsampler = upsamplers[i] 
#       
#      prev = F.interpolate(prev, scale_factor=0.5) 
#      prev = lrp_conv(upsampler.conv, a, prev)
#      # debug_relevance_scores(prev)
#      r.append(prev)
#
#  for index in reversed(range(len(resnets))):
#    # transformer
#    transformer = attentions[index]
#    rel, q, k, v, w = lrp_transformer(transformer, prev, time, text_embeddings, tog, res_weights, rho=rho)
#    queries.append(q)
#    keys.append(k)
#    values.append(v)
#    weights.append(w)
#
#    r += rel 
#    prev = r[-1]
#    # debug_relevance_scores(prev)
#    
#    # resnet
#    rel, time = lrp_resnetmid(resnets[index], a, prev, time, text_embeddings, tog, time, rho=rho)
#    r += rel
#    prev = r[-1]
#    
#    original_a_size = prev.shape[1] - down_samples[-(index + 1)].shape[1]
#    prev = prev[:, :original_a_size] 
#    r.append(prev)
#    # debug_relevance_scores(prev)
#    
#
#  return r, queries, keys, values, weights
#  
#  
#
#def lrp_mid(mid, a, prev, time, text_embeddings, tog, prev_time, res_weights, rho=reg_rho):  
#  r = [] 
#  res1, res2 = mid.resnets
#  transformer = mid.attentions[0]
#  
#  rel, prev_time = lrp_resnetmid(mid.resnets[1], a, prev, time, text_embeddings, tog, prev_time, rho=rho)
#  r+=rel
#  prev=r[-1]
#
#  # debug_relevance_scores(prev)
#  rel, queries, keys, values, weights = lrp_transformer(mid.attentions[0], prev, time, text_embeddings, tog, res_weights, rho=rho)
#  r += rel
#  prev = r[-1]  
#  # debug_relevance_scores(prev)
#
#  rel, prev_time = lrp_resnetmid(mid.resnets[0], a, prev, time, text_embeddings, tog, prev_time, rho=rho)
#  r+=rel
#  prev=r[-1]
#  # debug_relevance_scores(prev)
#  return r, prev_time, queries, keys, values, weights
#
#
#
#def lrp_down(layer, a, prev, time, text_embeddings, tog, prev_time, rho=reg_rho):  
#  r = []
#  for i in reversed(range(len(layer.resnets))):
#    resnet = layer.resnets[i]
#    rel, prev_time = lrp_resnetmid(resnet, a, prev, time, text_embeddings, tog, prev_time, rho=rho)
#    r+=rel
#    prev=r[-1]
#    # debug_relevance_scores(prev)
#    r += prev
#  return r, prev_time
#
#
#
#def lrp_crossdown(layer, a, prev, time, text_embeddings, tog, block, prev_time, res_weights, rho=reg_rho):  
#  r = []
#  queries = []
#  keys = []
#  values = []
#  weights = []
#  
#  
#
#
#  
#
#  for b in reversed(layer.downsamplers):
#    (name, a), l = next(tog)
#    
#    prev = lrp_conv_old(b.conv, a, prev)    
#    # debug_relevance_scores(prev)
#    r.append(prev)     
#    
#  
#  for i in reversed(range(len(layer.resnets))):
#    transformer = layer.attentions[i] 
#    resnet = layer.resnets[i]
#  
#    
#    rel, q, k, v, w = lrp_transformer(transformer, prev, time, text_embeddings, tog, res_weights, rho=rho)
#    queries.append(q)
#    keys.append(k)
#    values.append(v)
#    weights.append(v)
#    r += rel
#    prev = r[-1]
#    # debug_relevance_scores(prev)
#      
#
#    
#    rel, prev_time = lrp_resnetmid(resnet, a, prev, time, text_embeddings, tog, prev_time, rho=rho)
#    r+=rel
#    prev=r[-1]
#    # debug_relevance_scores(prev)
#  # print(len(queries))
#  return r, prev_time, queries, keys, values, weights
#
#
#
#
#  
#
## def forward_resnetblock(blocklist:ResnetBlock2D, input_tensor, temb):
##   resnet_block = blocklist
##   hidden_states = input_tensor
#
##   hidden_states = resnet_block.norm1(hidden_states)
##   hidden_states = resnet_block.nonlinearity(hidden_states)
#
##   hidden_states = resnet_block.conv1(hidden_states)
#
##   temb = resnet_block.nonlinearity(temb)
##   temb = resnet_block.time_emb_proj(temb)[:, :, None, None]
#
##   hidden_states = hidden_states + temb
#  
##   hidden_states = resnet_block.norm2(hidden_states)
##   hidden_states = resnet_block.nonlinearity(hidden_states)
#
##   hidden_states = resnet_block.dropout(hidden_states)
##   hidden_states = resnet_block.conv2(hidden_states)
#  
#
##   if resnet_block.conv_shortcut is not None:
##       input_tensor = resnet_block.conv_shortcut(input_tensor)
#
##   output_tensor = (input_tensor + hidden_states) / resnet_block.output_scale_factor
##   return output_tensor
#
#def lrp_resnetmid(resnet_block, a, prev, time, text_embeddings, tog, temb_prev, rho=reg_rho):
#  r = []
#  tembs = []
#  
#  
#  if resnet_block.conv_shortcut is not None:
#    (name2, conv2_a), conv2 = next(tog)  
#  (name3, dropout_a), dropout = next(tog)  
#
#  (name4, nonlin1_a), nonlin1 = next(tog)  
#  (name5, norm2_a), norm2 = next(tog)  
#
#  (name6, time_emb_proj_a), time_emb_proj = next(tog)  
#  (name7, nonlin2_a), nonlin2 = next(tog)  
#
#  (name8, conv1_a), conv1 = next(tog)  
#  (name9, nonlin3_a), nonlin3 = next(tog)  
#  (name10, norm1_a), norm1 = next(tog)  
#  (name11, input_tensor_a), input_tensor = next(tog)  
#  
#  if resnet_block.conv_shortcut is not None:
#    # print(norm1_a.shape)
#    input_activation = resnet_block.conv_shortcut(norm1_a)
#  else:
#    input_activation = norm1_a
#
#
#  # Reverse output scaling
#  conv2 = resnet_block.conv2(dropout_a)
#  prev, scale_prev = lrp_division((input_activation + conv2), resnet_block.output_scale_factor, prev)
#  prev, prev_input = lrp_addition(conv2, input_activation, prev)
#  
#  # conv2 
#  prev = lrp_conv(resnet_block.conv2, dropout_a, prev, rho=rho) 
#  r.append(prev)
#
#
#  prev = lrp_conv(nonlin1, norm2_a, prev, rho=rho)
#  r.append(prev)
#  
#  
#  prev = lrp_conv(resnet_block.norm2, conv1_a, prev, rho=rho)
#
#  temb = resnet_block.nonlinearity(time)
#  temb = resnet_block.time_emb_proj(temb)[:, :, None, None]
#  prev, prev_temb = lrp_addition(conv1_a, temb, prev)
#  
#  prev = lrp_conv(resnet_block.conv1, nonlin3_a, prev, rho=rho)
#  r.append(prev)
#  
#  # nonlin
#  prev = lrp_conv(resnet_block.nonlinearity, norm1_a, prev, rho=rho)
#  r.append(prev)
#
#  # norm1
#  prev = lrp_conv(resnet_block.norm1, norm1_a, prev, rho=rho)
#  r.append(prev)
#
#  return r, time
#
#
#
##   hidden_states = resnet_block.norm1(hidden_states)
##   hidden_states = resnet_block.nonlinearity(hidden_states)
#
##   hidden_states = resnet_block.conv1(hidden_states)
#
##   temb = resnet_block.nonlinearity(temb)
##   temb = resnet_block.time_emb_proj(temb)[:, :, None, None]
#
##   hidden_states = hidden_states + temb
#  
##   hidden_states = resnet_block.norm2(hidden_states)
##   hidden_states = resnet_block.nonlinearity(hidden_states)
#
##   hidden_states = resnet_block.dropout(hidden_states)
##   hidden_states = resnet_block.conv2(hidden_states)
#  
#
#
#
## def forward_transformer(transformer2D:Transformer2DModel, hidden_states, time, text_embeddings):
##   norm =  transformer2D.norm
##   proj_in = transformer2D.proj_in
#
##   # in transformer
##   transformer = transformer2D.transformer_blocks[0] 
##   proj_out = transformer2D.proj_out
##   residual = hidden_states
#  
##   hidden_states = norm(hidden_states)
##   hidden_states = proj_in(hidden_states)
#  
##   batch, _, height, width = hidden_states.shape
##   inner_dim = hidden_states.shape[1]
##   hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim) 
#
##   hidden_states = forward_attention(transformer, hidden_states, text_embeddings)
#  
##   hidden_states = (
##       hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
##   )
##   hidden_states = proj_out(hidden_states)
##   return hidden_states + residual
#
#def lrp_transformer(transformer_model, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
#  r = []
#  
#  proj_out = transformer_model.proj_out
#  transformer = transformer_model.transformer_blocks[0]
#  proj_in = transformer_model.proj_in
#  norm = transformer_model.norm
#  
#  (name1, proj_in_a), layer1 = next(tog)
#  # print(layer1)
#  # print(name1, proj_in_a.shape, layer1, "LAYER1")
#
#  with torch.no_grad():
#    attn_layers = [
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog),
#      next(tog), 
#      next(tog), 
#    ]
#
#  batch, _, height, width = prev.shape
#  inner_dim = prev.shape[1]
#  
#  (name, norm_a), layer = next(tog)
#  (name, conv_shortcut_a), layer = next(tog)
#  # print(name, norm_a.shape, layer, 'aneritns')
#
#  proj_in_a = (
#      proj_in_a.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
#  )
#  output = proj_out(proj_in_a)
#  prev, prev_norm = lrp_addition(output, conv_shortcut_a, prev)
#  # prev = prev - conv_shortcut_a
#  
#  # proj_out
#
#
#
#  prev = lrp_conv(proj_out, proj_in_a, prev, rho=rho)
#  r.append(prev)
#
#  # prev = prev.reshape(batch, height * width, inner_dim).permute(0, 2, 1)
#
#  # prev = prev.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
#  prev = prev.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim) 
#
#
#
#  # attention
#  
#  rel, queries, keys, values, weights = lrp_attention(transformer_model.transformer_blocks[0], prev, time, text_embeddings, iter(attn_layers), res_weights, rho=rho)
#  r += rel
#  prev = rel[-1]
#  # print(next(tog))
#
#  prev = prev.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
#  
#  # proj_in
#  
#  prev = lrp_conv(proj_in, norm_a, prev, rho=rho)
#  r.append(prev)
#
#
#  prev = lrp_conv(norm, conv_shortcut_a, prev, rho=rho)
#  r.append(prev)
#  del attn_layers
#  return r, queries, keys, values, weights
#
#
#
#
## def forward_attention(attn:BasicTransformerBlock, hidden_states, text_embeddings):
##   norm1 = attn.norm1
##   norm2 = attn.norm2
##   norm3 = attn.norm3
##   attn1 = attn.attn1
##   attn2 = attn.attn2
##   ff = attn.ff
#
##   norm_hidden_states = norm1(hidden_states)
##   attn_output = forward_selfattention(attn1, norm_hidden_states)
#
##   hidden_states = attn_output + hidden_states
#
##   norm_hidden_states = norm2(hidden_states) 
##   attn_output = forward_crossattention(attn2, norm_hidden_states, text_embeddings)
#  
##   hidden_states = attn_output + hidden_states
#
##   norm_hidden_states = norm3(hidden_states)
##   ff_output = ff(norm_hidden_states, encoder_hidden_states=text_embeddings)
##   hidden_states =  hidden_states + ff_output
#  
##   return hidden_states
#
#  
#def lrp_attention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
#  r = []
#  queries = []
#  keys = []
#  values = []
#  weights = []
#  
#  (name1, ff_a), layer1 = next(tog)
#  (name2, net2_a), layer = next(tog)
#  (name3, net1_a), layer = next(tog)
#  (name4, proj_a), layer = next(tog)
#  (name5, norm3_a), layer4 = next(tog)
#  (name6, to_out2_a), layer = next(tog) 
#
#  # (name7, to_out1_a), layer = next(tog)
#  # (name8, to_v_a), layer = next(tog)
#  # (name9, to_k_a), layer = next(tog)
#  # (name10, to_q_a), layer = next(tog)
#  # (name11, norm2_a), layer11 = next(tog)
#  
#  attn2_layers = [
#    next(tog),
#    next(tog),
#    next(tog),
#    next(tog),
#    next(tog)
#  ]
#  
#  (name12, to_out1_a), layer = next(tog)
#  # print(to_out1_a.shape, 'shape')
#  # (name13, to_out2_a), layer = next(tog)
#  # (name14, to_v_a), layer = next(tog)
#  # (name15, to_k_a), layer = next(tog)
#  # (name16, to_q_a), layer = next(tog)
#
#  attn1_layers = [
#    next(tog),
#    next(tog),
#    next(tog),
#    next(tog),
#    next(tog)
#  ]
#  
#  # (name17, norm1_a), layer = next(tog)
#  (name18, input_a), layer = next(tog)
#  # print(name1, ff_a.shape, layer1)
#  # print(name5, norm3_a.shape, layer4)
#  # print(name11, norm2_a.shape, layer11)
#  # print(name18, input_a.shape, layer)
#  # print('\n\n\n')
#  # print(name1)
#  # print(name2)
#  # print(name3)
#  # print(name4)
#  # print(name5)
#  # print(name6)
#  # print(name7)
#  # print(name8)
#  # print(name9)
#  # print(name10)
#  # print(name11)
#  # print(name12)
#  # print(name13)
#  # print(name14)
#  # print(name15)
#  # print(name16)
#  # print(name17)
#  # print(name18)
#  # print('\n\n\n')
#  
#  # print('\n\n\n\n\n')
#
#  # output=prev 
#  # Reverse feed-forward layer
#  # a = (norm3_a.data).requires_grad_(True)
#  # z = attn.ff(a, encoder_hidden_states=text_embeddings)
#  # # stabilizer = epsilon * torch.sign(z) + (z == 0).float()
#  # # s = (prev / (z + stabilizer)).data
#  # s = (prev / (z + epsilon)).data
#  # (z * s).sum().backward(retain_graph=True)
#  # c = a.grad
#  # prev = (a * c).data
#  prev = lrp_conv(attn.ff, norm3_a, prev, rho=rho)
#  r.append(prev)
#
#  # debug_relevance_scores(prev)
#  # print('\n')
#
#
#  prev = lrp_conv(attn.norm3, to_out2_a, prev, epsilon, rho=rho)
#  r.append(prev)
#
#  # debug_relevance_scores(prev)
#  # print('\n')
#
#
#
#
#  # Reverse addition (attn_output + hidden_states)
#  
#  # hidden_states = norm2_a
#  # relevance_attn_output = prev * (attn_output / (attn_output + hidden_states + epsilon))
#  # relevance_hidden_states = prev * (hidden_states / (attn_output + hidden_states + epsilon))
#  # prev = relevance_attn_output + relevance_hidden_states
#  # r.append(prev)
#
#
#
#  # Reverse cross-attention
#  
#  # Reverse self-attention
#  # a = (norm2_a.data).requires_grad_(True)
#  # z = sd.forward_crossattention(attn.attn2, a, text_embeddings)
#  # stabilizer = 1e1 * torch.sign(z) + (z == 0).float()
#  # s = (prev / (z+stabilizer)).data
#  # (z * s).sum().backward(retain_graph=True)
#  # c = a.grad
#  # prev = (a * c).data
#  # r.append(prev)
#
#  prev_q1, prev_k1, prev_v1, prev_weights1 = lrp_crossattention(attn.attn2, prev, time, text_embeddings, iter(attn2_layers), res_weights, rho=rho)
#  prev = prev_q1
#  r.append(prev)
#
#
#
#
#
#
#  # debug_relevance_scores(prev) 
#  # print('\n')
#  
#
#
#  prev = lrp_conv(attn.norm2, to_out1_a, prev, epsilon, rho=rho)  
#  r.append(prev)
#
#  # debug_relevance_scores(prev)
#  # print('\n')
#
#
#  # Reverse self-attention
#  # a = (norm1_a.data).requires_grad_(True)
#  # z = sd.forward_selfattention(attn.attn1, a)
#  # stabilizer = 1e1 * torch.sign(z) + (z == 0).float()
#  # s = (prev / (z+stabilizer)).data
#  # (z * s).sum().backward(retain_graph=True)
#  # c = a.grad
#  # prev = (a * c).data
#  # r.append(prev)
#
#  prev_q2, prev_k2, prev_v2, prev_weights2 = lrp_selfattention(attn.attn1, prev, time, text_embeddings, iter(attn1_layers), res_weights, rho=rho)
#  prev = prev_q2
#  r.append(prev)
#
#  # debug_relevance_scores(prev)
#  # print('\n')
#  
#  
#  batch, _, height, width = input_a.shape
#  inner_dim = input_a.shape[1]
#  input_a = input_a.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
#  # prev = prev.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
#  
#  # print(input_a.shape)
#  prev = lrp_conv(attn.norm1, input_a, prev, epsilon, rho=rho) 
#  r.append(prev)
#
#  # debug_relevance_scores(prev)
#  # print('\n\n\n\n\n')
#  
#  queries.append((prev_q1, prev_q2))
#  keys.append((prev_k1, prev_k2))
#  values.append((prev_v1, prev_v2))
#  weights.append((prev_weights1, prev_weights2))
#
#  return r, queries, keys, values, weights
#
#
#def lrp_selfattention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
#  (name7, to_out1_a), layer1 = next(tog)
#  (name8, to_v_a), layer2 = next(tog)
#  (name9, to_k_a), layer3 = next(tog)
#  (name10, to_q_a), layer4 = next(tog)
#  (name11, norm2_a), layer5 = next(tog)
#
#  inner_dim = to_k_a.shape[-1]
#  head_dim = inner_dim // attn.heads
#  batch_size, sequence_length, _ = norm2_a.shape
#
#  output = attn.to_out[1](to_out1_a)
#  prev, rescale_relevance = lrp_division(output, attn.rescale_output_factor, prev)
#  prev = lrp_conv(attn.to_out[1], to_out1_a, prev, rho=rho)
#
#  query = to_q_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#  key = to_k_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#  value = to_v_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#  scale_factor = 1 / math.sqrt(query.size(-1))
#  attn_weight1 = torch.matmul(query, key.transpose(-2, -1) * scale_factor) 
#  attn_weight2 = torch.softmax(attn_weight1, dim=-1) 
#  output = torch.matmul(attn_weight2, value).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#
#  prev = lrp_conv(attn.to_out[0], output, prev, rho=rho)
#  prev = prev.reshape(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)
#
#  R_query, R_key, R_value, R_weights = lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, (attn_weight1, attn_weight2))
#  
#  R_query = R_query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#  R_key = R_key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#  R_value = R_value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#  
#  prev_query = lrp_conv(attn.to_q, norm2_a, R_query, rho=rho)
#  prev_key = lrp_conv(attn.to_k, norm2_a, R_key, rho=rho)
#  prev_value = lrp_conv(attn.to_v, norm2_a, R_value, rho=rho)
#  
#  return prev_query, prev_key.detach().cpu(), prev_value.detach().cpu(), R_weights
#
#def lrp_crossattention(attn, prev, time, text_embeddings, tog, res_weights, rho=reg_rho):
#  (name7, to_out1_a), layer1 = next(tog)
#  (name8, to_v_a), layer2 = next(tog)
#  (name9, to_k_a), layer3 = next(tog)
#  (name10, to_q_a), layer4 = next(tog)
#  (name11, norm2_a), layer5 = next(tog)
#
#  inner_dim = to_k_a.shape[-1]
#  head_dim = inner_dim // attn.heads
#  batch_size, sequence_length, _ = norm2_a.shape
#
#  output = attn.to_out[1](to_out1_a)
#  prev, rescale_relevance = lrp_division(output, attn.rescale_output_factor, prev)
#  prev = lrp_conv(attn.to_out[1], to_out1_a, prev, rho=rho)
#  
#  query = to_q_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#  key = to_k_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#  value = to_v_a.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#  scale_factor = 1 / math.sqrt(query.size(-1))
#  attn_weight1 = torch.matmul(query, key.transpose(-2, -1) * scale_factor) 
#  attn_weight2 = torch.softmax(attn_weight1, dim=-1) 
#  output = torch.matmul(attn_weight2, value).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#
#  prev = lrp_conv(attn.to_out[0], output, prev, rho=rho)
#  prev = prev.reshape(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)
#
#  R_query, R_key, R_value, R_weights = lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, (attn_weight1, attn_weight2))
#
#  R_query = R_query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#  R_key = R_key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#  R_value = R_value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#    
#  prev_query = lrp_conv(attn.to_q, norm2_a, R_query, rho=rho)
#  prev_key = lrp_conv(attn.to_k, text_embeddings, R_key, rho=rho)
#  prev_value = lrp_conv(attn.to_v, text_embeddings, R_value, rho=rho)
#  
#  return prev_query, prev_key.detach().cpu(), prev_value.detach().cpu(), R_weights
#
#import math
#
#  
##def lrp_division(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)): 
##  # Ensure A and B are tensors
##  if not isinstance(A, torch.Tensor):
##    A = torch.tensor(A, dtype=torch.float32)
##  if not isinstance(B, torch.Tensor):
##    B = torch.tensor(B, dtype=torch.float32)
##
##  # Compute the output of the division
##  norm_B = B + epsilon * torch.where(B >= 0, torch.ones_like(B), -torch.ones_like(B))
##  z = A / (norm_B)
##  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
##
##
##  # Compute normalized relevance
##  relevance_norm = prev / (z)
##
##  # Distribute the relevance according to the contributions
##  relevance_a = relevance_norm * A
##  relevance_b = -relevance_norm * B * (A / (B ** 2))
##
##  return relevance_a, relevance_b
#
#def lrp_division(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)): 
#  # Ensure A and B are tensors
#  if not isinstance(A, torch.Tensor):
#    A = torch.tensor(A, dtype=torch.float32)
#  if not isinstance(B, torch.Tensor):
#    B = torch.tensor(B, dtype=torch.float32)
#
#  # Compute the output of the division
#  norm_B = B + epsilon * torch.where(B >= 0, torch.ones_like(B), -torch.ones_like(B))
#  z = A / (norm_B)
#  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#
#  term_a = A / norm_B
#  term_b = -A * B / (norm_B ** 2)
#
#  r_a = (term_a / z) * prev
#  r_b = (term_b / z) * prev
#  return r_a, r_b
#
#def lrp_multiplication(a, b, R_out):
#  # Compute the total contribution
#  z = a * b
#  total_contrib = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#
#  # Compute the individual contributions
#  contrib_a = (a * b) / total_contrib
#  contrib_b = (a * b) / total_contrib
#  # Distribute the relevance proportionally
#  R_a = (contrib_a / (contrib_a + contrib_b)) * R_out
#  R_b = (contrib_b / (contrib_a + contrib_b)) * R_out
#  return R_a, R_b  
#
#def lrp_addition(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)):
#  z = A + B
#  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#
#  relevance_norm = prev / z
#  relevance_a = relevance_norm * A
#  relevance_b = relevance_norm * B
#  return relevance_a, relevance_b
#
#def lrp_matmul(A, B, prev, incr=(lambda z: z + ((z>0).float()*2-1)*1e-1)): 
#  outputs = torch.matmul(A, B)   
#  z = outputs * 2
#  z = z + epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
#
#  relevance_norm = prev / z
#
#  relevance_a = torch.matmul(relevance_norm, B.transpose(-1, -2)).mul_(A)
#  relevance_b = torch.matmul(A.transpose(-1, -2), relevance_norm).mul_(B)
#  return relevance_a, relevance_b
#
#
##def lrp_softmax(z, s, R_s, epsilon=1e-9):
##    """
##    Propagate relevance through a softmax layer using a conservation-corrected rule.
##    
##    Given:
##      - z: the pre-softmax activations (logits). (Not used explicitly in this rule,
##           but available if you want to extend the rule.)
##      - s: the softmax outputs computed as s = softmax(z)
##      - R_s: the relevance scores at the softmax output.
##    
##    The idea is to redistribute R_s back to the logits in proportion to s.
##    However, a direct redistribution as s * R_s generally does not conserve relevance.
##    Therefore, we add a correction term such that:
##    
##         R_z = s * (R_s + correction)
##    
##    with correction = (R_s_total - (s * R_s).sum(dim=-1, keepdim=True)),
##    where R_s_total is the total relevance at the softmax output.
##    
##    This ensures that:
##         R_z.sum(dim=-1) = R_s_total
##    i.e. relevance is conserved.
##    
##    Parameters:
##      z (torch.Tensor): Pre-softmax activations (logits), shape [B, N].
##      s (torch.Tensor): Softmax outputs, shape [B, N].
##      R_s (torch.Tensor): Relevance at the softmax output, shape [B, N].
##      epsilon (float): Small constant for numerical stability (unused in this simple form).
##    
##    Returns:
##      torch.Tensor: Relevance scores for the logits, shape [B, N].
##    """
##    # Total relevance at the softmax output for each sample
##    R_s_total = R_s.sum(dim=-1, keepdim=True)
##    # Compute the partial relevance that would be assigned by a naive redistribution
##    naive = s * R_s
##    # The correction needed to ensure conservation:
##    correction = R_s_total - naive.sum(dim=-1, keepdim=True)
##    # Redistribute using softmax probabilities plus correction
##    R_z = s * (R_s + correction)
##    return R_z
#
#def lrp_softmax(inputs, output, output_relevance): 
#  with torch.no_grad():
#    inputs = torch.where(torch.isneginf(inputs), torch.tensor(0).to(inputs), inputs)
#    
#    relevance = inputs * (output_relevance - output * output_relevance.sum(-1, keepdim=True))
#    return relevance
#
#def lrp_scaled_dot_product_crossattention(attn, prev, query, key, value, to_out1_a, attn_weights):  
#  attn_weight1, attn_weight2 = attn_weights
#
#  scale_factor = 1 / math.sqrt(query.size(-1))
#  
#  output_rel = prev
#  R_weights, R_value = lrp_matmul(attn_weight2, value, output_rel)
#  
#  relevance_attn_weight = lrp_softmax(attn_weight1, attn_weight2, R_weights)
#  
#  R_query, R_key = lrp_matmul(query, key.transpose(-2, -1), relevance_attn_weight)
#  #del relevance_attn_weight
#  del attn_weight1, attn_weight2
#  return R_query.detach(), R_key.detach(), R_value.detach(), relevance_attn_weight.detach()
#
#
#
#
##def norm(relevance):
##  abs_relevance = relevance.abs()
##  min_val = abs_relevance.min()
##  max_val = abs_relevance.max()
##  normalized = (abs_relevance + min_val) / (max_val + min_val)
##  return normalized * torch.sign(relevance)
#
#
#def norm(relevance):
#  max_value = 1e10
#  # Replace positive and negative infinities with max_value and -max_value respectively
#  tensor = torch.where(torch.isinf(relevance), torch.sign(relevance) * max_value, relevance)
#  # Replace NaNs with zero
#  tensor = torch.where(torch.isnan(relevance), torch.zeros_like(relevance), relevance)
#  return tensor
#
#def norm_rel(relevance):
#  # Separate positive and negative relevance scores
#  positive_relevance = torch.clamp(relevance, min=0.0)
#  negative_relevance = torch.clamp(relevance, max=0.0).abs()
#
#  # Normalize positive relevance scores
#  if positive_relevance.max() > 0:
#      positive_norm = positive_relevance / positive_relevance.max()
#  else:
#      positive_norm = positive_relevance
#
#  # Normalize negative relevance scores
#  if negative_relevance.max() > 0:
#      negative_norm = negative_relevance / negative_relevance.max()
#  else:
#      negative_norm = negative_relevance
#
#  # Combine normalized scores with appropriate signs
#  normalized_relevance = positive_norm - negative_norm
#
#  return normalized_relevance
#
#def neg_heatmap(relevance, image, mult=1):
#  lrp_scores = relevance.unsqueeze(dim=-1)
#  scores = lrp_scores.sign()
#  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
#  scores = torch.cat((scores, scores, scores), dim=-1)
#
#  red = norm_rel(lrp_scores)*torch.Tensor([255*mult, 1, 1])
#  blue = norm_rel(-lrp_scores)*torch.Tensor([1, 1, 255*mult])
#
#
#  image1 = torch.where(scores>0, image, blue).to(torch.uint8).cpu()
#
#  plt.figure(figsize=(5,5))
#  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#  plt.axis('off')
#  plt.imshow(image1.numpy())
#  plt.show()
#
#
#def pos_heatmap(relevance, image, mult=1):
#  lrp_scores = relevance.unsqueeze(dim=-1)
#  scores = lrp_scores.sign()
#  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
#  scores = torch.cat((scores, scores, scores), dim=-1)
#
#  red = norm_rel(lrp_scores)*torch.Tensor([255*mult, 1, 1])
#  blue = norm_rel(-lrp_scores)*torch.Tensor([1, 1, 255*mult])
#
#
#  image1 = torch.where(scores>0, red, image).to(torch.uint8).cpu()
#
#
#  plt.figure(figsize=(5,5))
#  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#  plt.axis('off')
#  plt.imshow(image1.numpy())
#  plt.show()
#
#
#
#def comb_heatmap(relevance, mult=1):
#  lrp_scores = relevance.unsqueeze(dim=-1)
#  scores = lrp_scores.sign()
#  lrp_scores = torch.cat((lrp_scores, lrp_scores, lrp_scores), dim=-1)
#  scores = torch.cat((scores, scores, scores), dim=-1)
#
#  red = (norm_rel(lrp_scores))*torch.Tensor([255*mult, 1, 1])
#  blue = (norm_rel(-lrp_scores))*torch.Tensor([1, 1, 255*mult])
#
#  image1 = torch.where(scores>0, red, blue).to(torch.uint8).cpu()
#
#
#  plt.figure(figsize=(5,5))
#  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#  plt.axis('off')
#  plt.imshow(image1.numpy())
#  plt.show()
#
#
#
## def norm(relevance):
##   return ((relevance.abs() + relevance.abs().min()) / ((relevance).abs().max() + (relevance).abs().min())) * torch.sign(relevance)
#
#
## def normalize(gradients):
##   scale_factor=0.1
##   epsilon=1e1
##   max_grad = gradients.abs().max()
##   return gradients * (scale_factor / (max_grad + epsilon))
#
#
## def normalize(tens):
##   return tens / (tens.abs().max() + 1e-6) 
#
#def debug_relevance_scores(relevance):
#  print("Relevance Min:", relevance.min().item())
#  print("Relevance Max:", relevance.max().item())
#  print("Relevance Mean:", relevance.mean().item())
#  print("Relevance Std Dev:", relevance.std().item())
#    
#def visualize_relevance(decoded_relevance, original_image, save_path=None):
#    # Normalize the relevance maps
#    relevance_map = decoded_relevance.squeeze().cpu().numpy()
#    relevance_map = logarithmic_mapping(relevance_map)
#    
#    maxi = relevance_map.max()
#    mini = relevance_map.min()
#
#    # Overlay the relevance map on the original image
#    plt.figure(figsize=(10, 10))
#    plt.imshow(original_image)
#    plt.imshow(relevance_map,vmin=mini, vmax=maxi, cmap='jet', alpha=0.5)
#    plt.colorbar()
#    plt.title("Relevance Map Overlay")
#    
#    if save_path:
#        plt.savefig(save_path)
#    plt.show()
#
#
#  
#import numpy
#from matplotlib.colors import LogNorm, ListedColormap, Normalize, TwoSlopeNorm
#import matplotlib.cbook as cbook
#import matplotlib.colors as colors
#import cv2
#
#def overlay_attention_on_image(image, attn_weights, alpha=0.6, cmap='jet'):
#    """
#    Overlay a heatmap derived from attention weights on an image.
#    
#    Parameters:
#        image (np.array): The original image as an array of shape (H, W, 3) with values [0, 255].
#        attn_weights (torch.Tensor or np.array): A 2D attention map with shape (h, w).
#        alpha (float): Transparency for the heatmap overlay.
#        cmap (str): Name of the matplotlib colormap to use.
#        
#    Returns:
#        np.array: The image with the attention heatmap overlay.
#    """
#    # Convert attn_weights to numpy if it's a tensor
#    if isinstance(attn_weights, torch.Tensor):
#        attn_weights = attn_weights.detach().cpu().numpy()
#
#    # Normalize the attention weights to [0, 1]
#    attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)
#    
#    # Resize the attention map to the image dimensions
#    heatmap = cv2.resize(attn_norm, (image.shape[1], image.shape[0]))
#    print(heatmap)
#    
#    # Apply a colormap to the attention map to create an RGB heatmap
#    heatmap_color = plt.get_cmap(cmap)(heatmap)  # returns RGBA
#    heatmap_color = np.delete(heatmap_color, 3, axis=2)  # remove the alpha channel
#    
#    # Convert the image to float in [0, 1] if it's not already
#    if image.dtype == np.uint8:
#        image_float = image.astype(np.float32) / 255.0
#    else:
#        image_float = image
#
#    # Combine the heatmap with the image using alpha blending
#    overlay = np.concatenate(((heatmap_color * alpha), (image_float * (1 - alpha))))
#    overlay = np.clip(overlay, 0, 1)
#    
#    # Convert back to uint8 for display
#    overlay_uint8 = (overlay * 255).astype(np.uint8)
#    return overlay_uint8
#
##def heatmap(relevance, sx, sy, log=False, save_path='save.png'):
##    if log:
##        normalized_relevance = logarithmic_mapping(relevance)
##    else:
##        normalized_relevance = norm(relevance)
##
##    relevance_np = normalized_relevance.squeeze().cpu().numpy()
##
##    # Define color map
##    cmap = plt.cm.seismic
##    plt.figure(figsize=(sx, sy))
##    plt.imshow(relevance_np, cmap=cmap, interpolation='nearest')
##    plt.axis('off')
##    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
##    plt.show()
#
#def heatmap(R,sx,sy, log=False, save_path='save.png'):
#  if log:
#    R = logarithmic_mapping(R)
#    # b = 5*(numpy.abs(R)).mean()
#    # R = norm(R)
#    maxi = R.max()
#    mini = R.min()
#  else:
#    #mini = -8*((numpy.abs(R)**3.0).mean()**(1.0/3))
#    #maxi = 8*((numpy.abs(R)**3.0).mean()**(1.0/3))
#    R = norm_rel(R)
#    maxi = R.max()
#    mini = R.min()
#  
#
#  my_cmap = plt.cm.seismic
#  plt.figure(figsize=(sx,sy))
#  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#  plt.axis('off')
#  plt.imshow(R,cmap=my_cmap,interpolation='nearest')
#  plt_id = uuid.uuid4()
#  plt.savefig(save_path)
#
#  plt.show()
#def logarithmic_mapping(relevance_scores, scale_factor=1):
#    """
#    Apply a logarithmic transformation to LRP relevance scores to compress extreme values,
#    ensuring that large positive or negative scores do not dominate the overall contribution.
#
#    The function maps each score using the formula:
#        mapped_score = sign(score) * log(1 + abs(score) / scale_factor)
#    This transformation preserves the sign of the original score while compressing its magnitude.
#
#    Parameters:
#        relevance_scores (array-like): A list or NumPy array of relevance scores.
#        scale_factor (float): A constant used to adjust the scaling sensitivity. Default is 155.0.
#
#    Returns:
#        np.ndarray: A NumPy array of the logarithmically mapped relevance scores.
#    """
#    relevance_scores = np.array(relevance_scores)
#    mapped_scores = np.sign(relevance_scores) * np.log1p(np.abs(relevance_scores) / scale_factor)
#    return mapped_scores
#
#def logarithmic_mapping_torch(relevance_scores, scale_factor=1):
#    """
#    Apply a logarithmic transformation to LRP relevance scores using PyTorch,
#    compressing extreme values while preserving sign.
#
#    The transformation is:
#        mapped_score = sign(score) * log(1 + abs(score) / scale_factor)
#
#    Parameters:
#        relevance_scores (tensor or array-like): A tensor or array of relevance scores.
#        scale_factor (float): Scaling constant. Default is 1 (adjust as needed).
#
#    Returns:
#        torch.Tensor: A tensor of logarithmically mapped relevance scores.
#    """
#    if not isinstance(relevance_scores, torch.Tensor):
#        relevance_scores = torch.tensor(relevance_scores, dtype=torch.float32)
#    
#    mapped_scores = torch.sign(relevance_scores) * torch.log1p(torch.abs(relevance_scores) / scale_factor)
#    return mapped_scores
#
##def logarithmic_mapping(relevance_scores, scale_factor=155.0):
##    # Small epsilon to prevent log(0)
##
##    # Separate positive and negative relevance scores
##    positive_relevance = torch.clamp(relevance_scores, min=epsilon)
##    negative_relevance = torch.clamp(relevance_scores, max=epsilon).abs()
##
##    # Apply logarithmic scaling to positive relevance
##    positive_log = torch.log(positive_relevance)
##    if positive_log.max() > 0:
##        positive_norm = positive_log / positive_log.max()
##    else:
##        positive_norm = positive_log
##
##    # Apply logarithmic scaling to negative relevance
##    negative_log = torch.log(negative_relevance)
##    print(negative_log)
##
##    if negative_log.max() > 0:
##        negative_norm = negative_log / negative_log.max()
##    else:
##        negative_norm = negative_log
##
##    # Combine normalized logs with appropriate signs
##    negative_norm = torch.where(negative_norm <0, 0, negative_norm);
##    positive_norm = torch.where(positive_norm <0, 0, positive_norm);
##
##
##    normalized_relevance = positive_norm - negative_norm
##
##
##    # Apply scaling factor if desired
##    #normalized_relevance *= scale_factor
##
##    return normalized_relevance
#
##def logarithmic_mapping(relevance_scores, scale_factor=3):
##  sign = torch.sign(relevance_scores)
##  adjusted_scores = torch.where(relevance_scores == 0, epsilon, relevance_scores)
##  adjusted_scores =  torch.log(adjusted_scores.abs())
##  adjusted_scores += adjusted_scores.min().abs()
##  adjusted_scores *= sign
##  return adjusted_scores
#
#def save_tensor_to_mmap(tensor, filename):
#  np_tensor = tensor.cpu().numpy()
#  shape = np_tensor.shape
#  dtype = np_tensor.dtype
#
#  # Create a memory-mapped file with write access
#  mmap_file = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
#  mmap_file[:] = np_tensor[:]
#  mmap_file.flush()
#  del mmap_file  # Ensure changes are written to disk
#
## Function to load a tensor from a memory-mapped file
#def load_tensor_from_mmap(filename, shape, dtype):
#    # Open the memory-mapped file with read access
#    mmap_file = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
#    tensor = torch.tensor(mmap_file)
#    del mmap_file
#    return tensor
#
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
#
#def visualize_text_relevance(text_tokens, relevance_scores, cmap_name='coolwarm', save_path='save.png'):
#    # Normalize relevance scores between 0 and 1
#    relevance_scores = relevance_scores.cpu().numpy()
#    max_abs_score = np.max(np.abs(relevance_scores))
#    normalized_scores = relevance_scores;
#
#
#    # Create a color map
#    cmap = plt.get_cmap(cmap_name)
#
#    # Plot each token with its corresponding color
#    fig, ax = plt.subplots(figsize=(len(text_tokens) * 0.5, 1))
#    ax.axis('off')
#
#    for i, (token, score) in enumerate(zip(text_tokens, normalized_scores)):
#        color = cmap(score)
#        ax.text(i, 0, token, fontsize=12, color=color, ha='center', va='center', rotation=45)
#
#    # Adjust plot limits
#    ax.set_xlim(-0.5, len(text_tokens) - 0.5)
#    ax.set_ylim(-0.5, 0.5)
#
#    
#    file = open(f'{save_path.replace(".png", "")}_scores.txt', 'w+')
#    np.savetxt(f'{save_path.replace(".png", "")}_scores.txt', np.array(normalized_scores))
#    file.close()
#  
#    # Save and display the plot
#    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#    plt.show()
#
##def visualize_text_relevance(text_tokens, norm_scores, cmap_name='coolwarm', save_path='save.png'):
##  # Normalize the relevance scores to be between 0 and 1
##  # print(norm_scores)
##  sign = np.sign(norm_scores.cpu().numpy())
##  # norm_scores = logarithmic_mapping(norm_scores)
##
##  # norm_scores = norm(norm_scores)
##  
##  # Create a color map
##  cmap = plt.get_cmap(cmap_name)
##  
##  # Create a figure and axis
##  fig, ax = plt.subplots()
##  
##  # Plot each word with its corresponding color
##  new_scrores = []
##  for i, token in enumerate(text_tokens):
##    if token=='<|endoftext|>':
##      break;
##    if token=='<|startoftext|>':
##      continue;
##    new_scrores.append(norm_scores[i])
##    ax.text(i/5, 0.5, token, fontsize=12, color=cmap(norm_scores[i]), ha='center', va='center')
##  
##  # Remove the axis
##  ax.axis('off')
## 
##  # Show the plot
##  text_id = uuid.uuid4()
##  plt.savefig(save_path)
##  file = open(f'{save_path.replace(".png", "")}_scores.txt', 'w+')
##  np.savetxt(f'{save_path.replace(".png", "")}_scores.txt', np.array(new_scrores))
##  file.close()
##
##  plt.show()
##  
#
#def apply_lrp(unet, vae, layers, activations, samples, time, text_embeddings, initial_input, weights):
#  unet = unet
#  vae = vae
#  queries = []
#
#  keys = []
#  values = []
#  #attn_weights = []
#
#  temp = range(len(layers[5:563]))
#  L = len(temp)
#  lays = layers[5:563][::-1]
#  tog = zip(([None]+activations[5:562])[::-1], lays)
#
#  down_blocks = unet.down_blocks
#  mid_block = unet.mid_block
#  up_blocks = unet.up_blocks
#
#  conv_norm_out = unet.conv_norm_out
#  conv_act = unet.conv_act
#  conv_in = unet.conv_in
#
#  (name, a), layer = next(tog)
#  prev = activations[562][1]
#  basic_incr = lambda z: z+1e-9
#  prev = lrp_conv(layer, a, prev, incr=basic_incr)
#  
#
#  (name, a), layer = next(tog)
#  prev = lrp_conv(layer, a, prev, incr=basic_incr)
#
#
#  (name, a), layer = next(tog)
#  prev = lrp_conv(layer, a, prev, incr=basic_incr)
#  
#
#  down_block_samples = samples
#  res_samples = down_block_samples[0:3]
#  down_block_samples = down_block_samples[3:] 
#
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, q, k, v, w = lrp_crossup(up_blocks[-1], a, prev, res_samples, time, text_embeddings, tog, 3, res_weights, reg_rho)
#
#  prev = norm(r[-1])
#  del res_weights
#
#  queries += q
#  keys += k
#  values += v
#  #attn_weights += w
#
#  del r,q,k,v, w
#
#
#  res_samples = down_block_samples[0:3]
#  down_block_samples = down_block_samples[3:]
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, q, k, v, w = lrp_crossup(up_blocks[-2], a, prev, res_samples, time, text_embeddings, tog, 2, res_weights, reg_rho)
#  
#
#  queries += q
#  keys += k
#  values += v
#  #attn_weights += w
#
#  del res_weights
#  prev = norm(r[-1])
#  del r,q,k,v, w
#
#  res_samples = down_block_samples[0:3]
#  down_block_samples = down_block_samples[3:]
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, q, k, v, w = lrp_crossup(up_blocks[-3], a, prev, res_samples, time, text_embeddings, tog, 1, res_weights, reg_rho)
#  
#  queries += q
#  keys += k
#  values += v
#  #attn_weights += w
#
#  del res_weights
#  prev = norm(r[-1])
#  del r,q,k,v, w
#
#
#
#  res_samples = down_block_samples[0:3]
#  down_block_samples = down_block_samples[3:]
#  r = lrp_up(up_blocks[0], a, prev, res_samples, time, text_embeddings, tog, 1, reg_rho)
#  prev = norm(r[-1])
#  del r
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, time_prev, q, k, v, w = lrp_mid(mid_block, a, prev, time, text_embeddings, tog, time, res_weights,  rho=gamma_rho)
#   
#  queries.append(q)
#  keys.append(k)
#  values.append(v)
#  #attn_weights.append(w)
#
#  del res_weights
#  prev = norm(r[-1])
#  del r,q,k,v,w
#
#
#  r, time_prev = lrp_down(down_blocks[-1], a, prev, time, text_embeddings, tog, time_prev,  rho=gamma_rho)
#  prev = norm(r[-1])
#  del r
#
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-2], a, prev, time, text_embeddings, tog, 2, time_prev, res_weights,  rho=gamma_rho)
#
#  queries += q
#  keys += k
#  values += v
#  #attn_weights += w
#
#  del res_weights
#  prev = norm(r[-1])
#  del r,q,k,v,w
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-3], a, prev, time, text_embeddings, tog, 1, time_prev, res_weights, rho=gamma_rho)
#  
#
#  queries += q
#  keys += k
#  values += v
#  #attn_weights += w
#
#  del res_weights
#  prev = norm(r[-1])
#  del r,q,k,v
#
#
#  res_weights = weights[-2:]
#  weights = weights[:-2]
#  r, time_prev, q, k, v, w = lrp_crossdown(down_blocks[-4], a, prev, time, text_embeddings, tog, 0, time_prev, res_weights, rho=gamma_rho)
#  del res_weights
#  prev = norm(r[-1])
#  queries += q
#  keys += k
#  #attn_weights += w
#
#  values += v
#  del r,q,k,v, w
#
#  r = lrp_conv(conv_in, initial_input, prev, rho=gamma_rho)
#  del tog
#
#  return r, queries, keys, values, [] #attn_weights
#
