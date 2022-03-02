import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np 

class Model(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers, isCuda=True, **kwargs):
		super().__init__()

		self.encoder = EncoderRNN(in_channels, out_channels, num_layers, isCuda)
		self.drop_en = nn.Dropout(p=0.5)
		self.bn2 = nn.BatchNorm1d(out_channels*30)
		self.fc = nn.Linear(out_channels*30, out_channels)




	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, x):
		'''
		Args:
			x: (batch, time_step, input_size)
		Returns:
			num_output size
		'''
		in_data = x 
		encoded_output, hidden = self.encoder(in_data)

		# fc_input = self.bn2(last_tensor)
		out = self.fc(hidden[-1])
		return out

if __name__ == '__main__':
	model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)
