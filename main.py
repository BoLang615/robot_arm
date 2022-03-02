import argparse
import os 
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
from dataloader import DataLayer, build_data_loader
#import torchsnooper
import debugpy
debugpy.listen(5679)
debugpy.wait_for_client()

classes = {0: 'drawer',
           1: 'pick',
           2: 'stir'}


def train(model, train_gen, criterion, optimizer, args):
	model.train() # Sets the module in training mode.
	count = 0
	total_loss = 0

	loader = tqdm(train_gen, total=len(train_gen))
	with torch.set_grad_enabled(True):
		for batch_idx, data in enumerate(loader):
			action_lenth = data['action_lenth']
			assert torch.unique(action_lenth).shape[0] == 1
			batch_size = data['input_x'].shape[0]
			count += batch_size
			
			input_x = data['input_x'].to(args.device)
			
			target_y = data['target_y'].to(args.device)
			# target_bbox_st = data['target_y_st'].to(device)

			
			output = model(input_x)
			loss = criterion(output, target_y)
			loader.set_description(f"Loss: {loss.item():.4f}")


			total_loss += loss.item()* batch_size
			
			# optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
	total_loss /= count
	
	return total_loss

def val(model, val_gen, criterion, args):
	model.eval() # Sets the module in val mode.
	count = 0
	total_loss = 0
	correct = 0
	total = 0
	loader = tqdm(val_gen, total=len(val_gen))
	with torch.set_grad_enabled(False):
		for batch_idx, data in enumerate(loader):
			action_lenth = data['action_lenth']
			assert torch.unique(action_lenth).shape[0] == 1
			batch_size = data['input_x'].shape[0]
			count += batch_size
			
			input_x = data['input_x'].to(args.device)
			
			target_y = data['target_y'].to(args.device)
			# target_bbox_st = data['target_y_st'].to(device)
			seq_info = data['seq_info']
			
			output = model(input_x)
			loss = criterion(output, target_y)
			loader.set_description(f"Loss: {loss.item():.4f}")
			total += target_y.size(0)
			_, predicted = torch.max(output, 1)
			correct += (predicted == target_y).sum().item()

			print(f'Predicted: {seq_info[0]}', 'Action: ',' '.join(f'{classes[int(predicted[j])]:5s}'
										for j in range(batch_size)))
			total_loss += loss.item()* batch_size
	print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))		


		
	total_loss /= count
	


	return total_loss



            

   

def main(args):
	this_dir = osp.dirname(__file__)

	save_dir = osp.join(this_dir, 'checkpoints')
	if not osp.isdir(save_dir):
		os.makedirs(save_dir)

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
 
	if osp.isfile(args.checkpoint):
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		args.start_epoch += checkpoint['epoch']
		del checkpoint


	criterion = nn.CrossEntropyLoss().to(args.device)

	train_loader = build_data_loader(args, 'train', batch_size = 1)
	val_loader = build_data_loader(args, 'test', batch_size = 1)

	print("Number of train samples:", train_loader.__len__())
	print("Number of test samples:", val_loader.__len__())
 
	# train
	min_loss = 1e6

	best_model = None



	for epoch in range(args.start_epoch, args.epochs+args.start_epoch):

		train_loss = train(model, train_loader, criterion, optimizer, args)

		print('Train Epoch: {} \t  Loss: {:.6f}'.format(
				epoch, train_loss))



		# val
		test_loss = val(model, val_loader, criterion, args)
		# lr_scheduler.step(val_loss)



		if test_loss < min_loss:
			try:
				os.remove(best_model)
			except:
				pass

			min_loss = test_loss
			saved_model_name = 'epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%min_loss + '.pth'

			print("Saving checkpoints: " + saved_model_name )
			if not os.path.isdir(save_dir):
				os.mkdir(save_dir)

			save_dict = {   'epoch': epoch,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict()}
			torch.save(save_dict, os.path.join(save_dir, saved_model_name))
			best_model = os.path.join(save_dir, saved_model_name)





if __name__ == '__main__':
    main(args)	