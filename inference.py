import os.path as osp
import sys
import numpy as np 
import torch
import torch.optim as optim
from torch import nn
from model import Model


import random
from tqdm import tqdm
from argument_parser import args
from testloader import data_loader
#import torchsnooper
import debugpy
debugpy.listen(5679)
debugpy.wait_for_client()

classes = {0: 'drawer',
           1: 'pick',
           2: 'stir'}
log = 'test_log.txt'


def inference(model, val_gen, criterion, args):
	model.eval() # Sets the module in training mode.
	count = 0

	loader = tqdm(val_gen, total=len(val_gen))
	current_scene = 0
	sliding_window_index = 0
	content = []
	with torch.set_grad_enabled(False):
		for batch_idx, data in enumerate(loader):
			
			batch_size = data['input_x'].shape[0]
			count += batch_size
			
			input_x = data['input_x'].to(args.device)
			
			target_y = data['target_y'].to(args.device)
			# target_bbox_st = data['target_y_st'].to(device)
			seq_info = data['seq_info']

			output = model(input_x)


			_, predicted = torch.max(output, dim=1)
			predict_class = classes[int(predicted[0])]

			end_idx = sliding_window_index+args.sliding_window-1
			groud_truth = target_y[0].detach().cpu().numpy()
			if len(data['split_index'][0]) == 1 and data['split_index'][0].item() == 20:
				groud_truth = classes[int(groud_truth)]
			print(f'Predicted: {seq_info[0]},', f'sliding_window: {sliding_window_index}-{end_idx},', 'Predict Action: ',' '.join(f'{classes[int(predicted[j])]:5s}'
										for j in range(batch_size)), ' ', f'Groud truth: {groud_truth}')
			trajectory_line = "Predicted: {}, sliding_window {}-{}, Predict Action: {}, Groud truth: {}\n".format(seq_info[0], sliding_window_index, end_idx, classes[int(predicted[0])], groud_truth)
			content.append(trajectory_line)
			if current_scene == int(seq_info[0].rstrip('.txt').split('/')[-1]):
				sliding_window_index += 1
			else:
				sliding_window_index = 0
				current_scene += 1
	if args.save_prediction:
		with open(log, 'w') as f:
			f.writelines(content)	




            

   

def main(args):
	if not torch.cuda.is_available() or args.device == 'cpu':
		args.device = torch.device('cpu')
	else:
		if torch.cuda.device_count() == 1:
			# If you have CUDA_VISIBLE_DEVICES set, which you should,
			# then this will prevent leftover flag arguments from
			# messing with the device allocation.
			args.device = 'cuda:0'

		args.device = torch.device(args.device)
	torch.cuda.set_device(args.device)
	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(args.seed)
   
	model = Model(in_channels=17, out_channels=args.classes, num_layers=1)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
	# 														min_lr=1e-10, verbose=1)
	model = model.to(args.device)
	criterion = nn.CrossEntropyLoss().to(args.device)

	if osp.isfile(args.checkpoint):
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		print('Successfull loaded from {}'.format(args.checkpoint))
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		args.start_epoch += checkpoint['epoch']
		del checkpoint



	test_root = './dataset/test/drawerpickstir40'

	test_loader = data_loader(args, test_root, pra_batch_size=1)


	print("Number of test sliding windows:", test_loader.__len__())



	inference(model, test_loader, criterion, args)








if __name__ == '__main__':
    main(args)	