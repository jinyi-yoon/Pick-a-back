import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch import nn, einsum
from einops import rearrange, repeat
import pdb
from pprint import pprint

DEFAULT_THRESHOLD = 5e-3

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class SharableConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(SharableConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        # Give real-valued mask weights per task to manage the shared part from previous tasks.
        self.piggymask = None

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            # print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, layer_info=None, name=None):
        if self.piggymask is not None:
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            # Mask weights with above mask.
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight

        # Perform conv using modified weight.
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)

class SharableLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(SharableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.piggymask = None

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        if self.piggymask is not None:
            # pdb.set_trace()
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            # Mask weights with above mask.
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight
        # Get output using modified weight.
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None:
            self.bias.data = fn(self.bias.data)

class SharableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 mask_init='1s', mask_scale=1e-2, 
                 threshold_fn='binarizer', threshold=None):
        super(SharableMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 각 head의 차원
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        self.q_proj = SharableLinear(embed_dim, embed_dim, bias=False, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)
        self.k_proj = SharableLinear(embed_dim, embed_dim, bias=False, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)
        self.v_proj = SharableLinear(embed_dim, embed_dim, bias=False, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)
        self.out_proj = SharableLinear(embed_dim, embed_dim, bias=True, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)
        self.Dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot-product

        # Query, Key, Value projection layers
        # self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.out_proj = nn.Linear(embed_dim, embed_dim)  # 최종 출력 프로젝션

        # self.dropout = nn.Dropout(dropout)
        # self.scale = self.head_dim ** -0.5  # Scaling factor for dot-product

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (seq_len, batch_size, embed_dim)
        mask: (batch_size, seq_len, seq_len), optional
        """
        B, N, D = query.shape  # (seq_len, batch_size, embed_dim)
        H = self.num_heads
        head_dim = self.head_dim

        # 1. Query, Key, Value를 각각 Projection
        q = self.q_proj(query)  # (N, B, D)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Multi-head 차원 변환: (batch, num_heads, seq_len, head_dim)
        q = rearrange(q, "n b (h d) -> b h n d", h=H)  # (B, H, N, head_dim)
        k = rearrange(k, "n b (h d) -> b h n d", h=H)
        v = rearrange(v, "n b (h d) -> b h n d", h=H)

        # 3. Scaled Dot-Product Attention 수행
        attn_scores = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale  # (B, H, N, N)

        # 4. Masking 처리 (선택적)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_weights = self.Dropout(attn_weights)

        # 5. Attention 적용 후 값 추출
        attn_output = einsum("b h i j, b h j d -> b h i d", attn_weights, v)  # (B, H, N, head_dim)

        # 6. 원래 차원으로 되돌리기
        attn_output = rearrange(attn_output, "b h n d -> n b (h d)")  # (N, B, D)

        # 7. 최종 Linear Projection 후 반환
        return self.out_proj(attn_output), attn_weights

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SharableFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        self.net = nn.Sequential(
            SharableLinear(dim, dim * mult * 2),
            GEGLU(),
            SharableLinear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SharableAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None):
        super().__init__()
        inner_dim = dim_head *heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = SharableLinear(query_dim, inner_dim, bias=False)
        self.to_kv = SharableLinear(context_dim, inner_dim * 2, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = SharableLinear(inner_dim, query_dim)
    
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask): 
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)