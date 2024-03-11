import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import numpy as np

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM

class NMS(nn.Module):
    def __init__(self, r_first = 1, r_second = 4, m = 1.01, s = 5, loss_coef = 1, **kwargs):
        super(NMS, self).__init__()
        self.m = m
        self.loss_coef = loss_coef
        self.r_first = r_first
        self.r_second = r_second
        self.s = s
        
        p = 12 / r_first / (r_first + 2) - 2
        f_first = np.array([[1, p, 1]]) / (2 + p)

        weight_1_1 = torch.from_numpy(np.ascontiguousarray(f_first[::-1, ::-1])).float()
        weigth_1_1_t = torch.from_numpy(np.ascontiguousarray(f_first[::-1, ::-1]).T).float()
        self.conv_1_1 = nn.Conv2d(1, 1, (1, 3), padding='same', bias=False)
        self.conv_1_1_t = nn.Conv2d(1, 1, (3, 1), padding='same', bias=False)
        self.conv_1_1.weight = nn.Parameter(weight_1_1.unsqueeze(0).unsqueeze(0))
        self.conv_1_1_t.weight = nn.Parameter(weigth_1_1_t.unsqueeze(0).unsqueeze(0))
      
        f_second = np.array([list(range(1, r_second + 1)) + [r_second + 1] + list(range(r_second, 0, -1))]) / (r_second + 1) ** 2        
        weighth_2_1 = torch.from_numpy(np.ascontiguousarray(f_second[::-1, ::-1])).float()
        weigth_2_1_t = torch.from_numpy(np.ascontiguousarray(f_second[::-1, ::-1]).T).float()

        self.conv_2_1 = nn.Conv2d(1, 1, (1, 9), padding='same', bias=False)
        self.conv_2_1_t = nn.Conv2d(1, 1, (9, 1), padding='same', bias=False)
        self.conv_2_1.weight = nn.Parameter(weighth_2_1.unsqueeze(0).unsqueeze(0))
        self.conv_2_1_t.weight = nn.Parameter(weigth_2_1_t.unsqueeze(0).unsqueeze(0))
        
        for param in self.parameters():
            param.requires_grad = False
            
         
    def interp(self,edge, x, y):
        """Forward function of PSP module."""
        x0 = x.int()
        y0 = y.int()
        x1 = x0 + 1
        y1 = y0 + 1
        
        dx0 = x - x0
        dy0 = y - y0
        dx1 = 1 - dx0
        dy1 = 1 - dy0
        
        first_val = torch.reshape(edge[0,0,torch.ravel(y0).tolist(),torch.ravel(x0).tolist()],(1,1,edge.size(2),edge.size(3)))
        second_val = torch.reshape(edge[0,0,torch.ravel(y0).tolist(),torch.ravel(x1).tolist()],(1,1,edge.size(2),edge.size(3)))
        third_val = torch.reshape(edge[0,0,torch.ravel(y1).tolist(),torch.ravel(x0).tolist()],(1,1,edge.size(2),edge.size(3)))
        fourth_val = torch.reshape(edge[0,0,torch.ravel(y1).tolist(),torch.ravel(x1).tolist()],(1,1,edge.size(2),edge.size(3)))
        out = first_val * dx1 * dy1 + second_val * dx0 * dy1 + third_val * dx1 * dy0 + fourth_val * dx0 * dy0
        return out

    def forward(self, edge):
        print(torch.amax(edge))
        print(torch.amin(edge),flush=True)
        print("IN NMS", flush=True)
        _,_,h,w = edge.size()
        edge = self.conv_1_1_t(self.conv_1_1(edge))
        a = self.conv_2_1_t(self.conv_2_1(edge)).squeeze()
        oy, ox = torch.gradient(a)
        _, oxx = torch.gradient(ox)
        oyy, oxy = torch.gradient(oy)
        ori = torch.remainder(torch.arctan(oyy * torch.sign(-oxy) / (oxx + 1e-5)), 3.141592653589793)
        #print(ori, "adsad")
        """Forward function."""
        out = torch.clone(edge)
        e = torch.clone(edge)        
        ori = ori.unsqueeze(0).unsqueeze(0)
        cos_arr = torch.cos(ori)
        sin_arr = torch.sin(ori)
        
        #y,x = np.indices((h,w))
        
        y = torch.arange(h).unsqueeze(1).expand(h, w).cuda()
        x = torch.arange(w).unsqueeze(0).expand(h, w).cuda()
        d2_interp_x = 2 * cos_arr + x
        d_2_interp_x = -2 * cos_arr + x
        d1_interp_x =  cos_arr + x
        d_1_interp_x = -1 * cos_arr + x   
        
        d2_interp_y = 2 * sin_arr + y
        d_2_interp_y = -2 * sin_arr + y
        d1_interp_y =  sin_arr + y
        d_1_interp_y = -1 * sin_arr + y 
        
        #clear boundaries
        d2_interp_x[d2_interp_x<0] = 0    
        d_2_interp_x[d_2_interp_x<0] = 0
        d1_interp_x[d1_interp_x<0 ] = 0
        d_1_interp_x[d_1_interp_x<0] = 0
        
        d2_interp_y[d2_interp_y<0] = 0
        d_2_interp_y[d_2_interp_y<0] = 0
        d1_interp_y[d1_interp_y<0 ] = 0
        d_1_interp_y[d_1_interp_y<0] = 0
        
        d2_interp_x[d2_interp_x > w - 1.001] = w - 1.001
        d_2_interp_x[d_2_interp_x > w - 1.001] = w - 1.001
        d1_interp_x[d1_interp_x > w - 1.001 ] = w - 1.001
        d_1_interp_x[d_1_interp_x > w - 1.001] = w - 1.001
        
        d2_interp_y[d2_interp_y > h - 1.001] = h - 1.001
        d_2_interp_y[d_2_interp_y> h - 1.001] = h - 1.001
        d1_interp_y[d1_interp_y> h - 1.001 ] = h - 1.001
        d_1_interp_y[d_1_interp_y > h - 1.001] = h -1.001

        
        e0_1 = self.interp(edge, d2_interp_x, d2_interp_y) 
        e0_2 = self.interp(edge, d1_interp_x, d1_interp_y)
        e0_3 = self.interp(edge, d_2_interp_x, d_2_interp_y)
        e0_4 = self.interp(edge, d_1_interp_x, d_1_interp_y)
        
        e *= self.m
        suppresed_1 = torch.logical_or(e < e0_1, e < e0_2)
        suppresed_2 = torch.logical_or(e < e0_3, e < e0_4)
        suppresed = torch.logical_or(suppresed_1,suppresed_2)
        
            
        out *= (1-suppresed.long())
        gradient = torch.clone(1-suppresed.long())
        #gradient[gradient == 1 ] = self.loss_coef
        
        return out,gradient

@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        #self.NMS = NMS().cuda()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.last_norm = nn.GroupNorm(1, 1)
        self.counter = 0
        self.initial = 4
        self.last = 0.5

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] =laterals[i - 1] +  resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        #print(output.size())
        #print(output)
        #print("---")
        #output,grad = self.NMS(torch.sigmoid(output))
        #print(output.size(), flush=True)
        #output /=  (self.initial - ( (self.initial - self.last) * (self.counter // 50000) / 4.))
        #self.counter +=1
        return torch.sigmoid(output)
        #if nn.Module().training:
        #    return output,grad
        #else:
        #    return output
        #print(torch.amin(output), flush=True)
        #return self.last_norm(output)
        #return output
        #return torch.sigmoid(output)
