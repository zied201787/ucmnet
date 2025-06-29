import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['XNORConv2d', 'XNORLinear', 'BNNConv', 'BNNLinear', 'BiRealConv']

class BinActive(Function):
    '''
    Binarize the input activations for BNN and XNOR networks.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinActiveBiReal(nn.Module):
    '''
    Binarize the input activations for BiReal networks.
    '''
    def __init__(self):
        super(BinActiveBiReal, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        return out_forward.detach() - out3.detach() + out3

class BiRealConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(BiRealConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001, 
                                 requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True), 
                                     dim=2, keepdim=True), 
                          dim=1, keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        bx = BinActiveBiReal()(x)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        bw = w_alpha.detach() - cliped_bw.detach() + cliped_bw
        return F.conv2d(bx, bw, self.bias, self.stride, self.padding)

class XNORConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, dropout_ratio=0):
        super(XNORConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride, padding, dilation, groups, bias)
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels // groups, 
                                             kernel_size, kernel_size)) * 0.001,
                                  requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        if not bias:
            self.register_parameter('bias', None)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True), 
                          dim=2, keepdim=True), dim=1, keepdim=True).detach()
        bw = BinActive().apply(w)
        bx = BinActive().apply(x)
        bw = bw * alpha
        return F.conv2d(bx, bw, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups)

class BNNConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(BNNConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,
                                  requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bw = BinActive().apply(self.weight)
        bx = BinActive().apply(x)
        return F.conv2d(bx, bw, self.bias, self.stride, self.padding)

class BNNLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=False):
        super(BNNLinear, self).__init__(in_channels, out_channels, bias)

    def forward(self, x):
        bw = BinActive().apply(self.weight)
        bx = BinActive().apply(x)
        return F.linear(bx, bw, self.bias)

class XNORLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(XNORLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(abs(w), dim=1, keepdim=True).detach()
        bw = BinActive().apply(w)
        bx = BinActive().apply(x)
        bw = bw * alpha
        return F.linear(bx, bw, self.bias)