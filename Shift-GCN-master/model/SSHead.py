from torch import nn
from .ttt_shift_gcn import *
# from torchsummary import summary

import math
import copy


class ExtractorHead(nn.Module):
	def __init__(self, ext, head, bn_layer):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head
		self.bn_layer = bn_layer

	def forward(self, x):
		N, C, T, V, M = x.size()
		x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
		x = self.bn_layer(x)
		x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
		x = self.ext(x)

		c_new = x.size(1)
		x = x.view(N, M, c_new, -1)
		x = x.mean(3).mean(1)
		return self.head(x)


class ExtraHead(nn.Module):
	def __init__(self, in_channels=256, aux_num_class=2, num_point=17, num_person=2):
		super(ExtraHead, self).__init__()
		self.in_channels = in_channels
		self.out_channels = aux_num_class
		self.fc1 = nn.Linear(self.in_channels, self.out_channels)
		# self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
		
		nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / self.out_channels))
		# bn_init(self.data_bn, 1)

	def forward(self, x):
		# print(x.shape)
		x = self.fc1(x)
		# x = self.data_bn(x)
		return x

class MainTrackHead(nn.Module):
	"""docstring for MainTrackHead"""
	def __init__(self, in_channels=256, num_class=155):
		super(MainTrackHead, self).__init__()
		self.in_channels = in_channels
		self.out_channels = num_class

		self.fc2 = nn.Linear(256, self.out_channels)
		# self.data_bn2 = nn.BatchNorm1d(self.out_channels)
		nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / self.out_channels))
		# bn_init(self.data_bn2, 1)

	def forward(self, x):
		x = self.fc2(x)
		# x = self.data_bn2(x)
		return x

def extractor_from_layer10(net):
	bn_layer = [net.data_bn]
	layers = [net.l1, net.l2, net.l3, net.l4, net.l5, net.l6, net.l7, net.l8, net.l9, net.l10]
	return nn.Sequential(*bn_layer), nn.Sequential(*layers)

def build_model(args, devices, in_channels=3, num_class=155, aux_num_class=2):
	net = BaseModel(**args.model_args).cuda(devices)
	aux_head = ExtraHead().cuda(devices)
	cls_head = MainTrackHead().cuda(devices)
	return net, aux_head, cls_head