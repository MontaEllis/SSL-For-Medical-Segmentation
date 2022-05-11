import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GaborConv3d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, device="cpu", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(GaborConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _triple(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta_1 = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.theta_2 = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.theta_3 = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 3]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 3]))[0]
        self.z0 = torch.ceil(torch.Tensor([self.kernel_size[2] / 3]))[0]
        self.device = device

    def forward(self, input_image):
        x, y, z = torch.meshgrid([torch.linspace(-self.x0, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0, self.y0, self.kernel_size[1]),
                               torch.linspace(-self.z0, self.z0, self.kernel_size[2])])
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta_1 = self.theta_1[i, j].expand_as(y)
                theta_2 = self.theta_2[i, j].expand_as(y)
                theta_3 = self.theta_3[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)


                # % yaw - rotation about z
                # xp = x*cos(theta)+y*sin(theta);
                # yp = -x*sin(theta)+y*cos(theta);
                # zp = z;
                # % pitch - rotation about y
                # xp = xp*cos(phi)-zp*sin(phi);
                # yp = yp;
                # zp = xp*sin(phi)+zp*cos(phi);
                # % % roll - rotation about x - unimplemented, requires "roll" angle
                # % xp = xp;
                # % yp = yp*cos(roll) + zp*sin(roll);
                # % zp = -yp*sin(roll) + zp*cos(roll);


                rotx = x * torch.cos(theta_1) + y * torch.sin(theta_1)
                roty = -x * torch.sin(theta_1) + y * torch.cos(theta_1)
                rotz = z
                x_ = rotx
                y_ = roty
                z_ = rotz

                rotx = x_ * torch.cos(theta_2) - z_ * torch.sin(theta_2)
                roty = y_
                rotz = x_ * torch.sin(theta_2) + z_ * torch.cos(theta_2)
                x_ = rotx
                y_ = roty
                z_ = rotz

                rotx = x_
                roty = y_ * torch.cos(theta_3) + z_ * torch.sin(theta_3)
                rotz = -y_ * torch.sin(theta_3) + z_ * torch.cos(theta_3)


                g = torch.zeros(y.shape)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2 + rotz ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv3d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


    
class GaborNN(nn.Module):
    def __init__(self,band,classes):
        super(GaborNN, self).__init__()
        self.name = 'GaborNN'
        self.g0 = GaborConv3d(in_channels=band, out_channels=96, kernel_size=3, device=device)
        self.m1 = nn.MaxPool2d(kernel_size=(1,1))
        self.c1 = nn.Conv2d(96, 128, (1,1))
        self.m2 = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(128*2*2, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        # x = x.permute(0,3,1,2)
        x = self.g0(x)
        
        return x


  
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = x.cuda()
    print("x size: {}".format(x.size()))

    model = GaborNN(1,16).to(device)

    out = model(x)
    print("out size: {}".format(out.size()))