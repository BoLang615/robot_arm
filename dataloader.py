
import os
import sys

import numpy as np
import torch
import glob
from torch.utils import data
from argument_parser import args
import json
import random
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DataLayer(data.Dataset):

    def __init__(self, args, split):
        
        self.dataroot = args.dataroot
        self.split = split
        self.batch_size = args.batch_size if split != 'test' else 1
        self.downsample_rate = args.downsample_rate
        self.sliding_window_size = 10
        self.action = {'drawer': 0,
                       'pick': 1,
                       'stir': 2,
                        }
        self.samples = []

        for action in self.action.keys():
            
            action_list = sorted(glob.glob(os.path.join(self.dataroot, self.split, '{}/*.txt'.format(action))))   
            self.samples.extend(action_list) 


        self.labels = [self.action[k.split('/')[-2]] for k in self.samples]
            
        self.len_dict = {}
        for i in range(len(self.samples)):
            downsample_content = self.load_seq_data(self.samples[i])
            action_lenth = downsample_content.shape[0]
            if action_lenth not in self.len_dict:
                self.len_dict[action_lenth] = []
            self.len_dict[action_lenth].append(i)
        self.shuffle_dataset()
        
    def shuffle_dataset(self):
        self._init_inputs()
        
    def _init_inputs(self):
        '''
        shuffle the data based on its length
        '''
        self.inputs = []
        
        for length in self.len_dict:
            indices = self.len_dict[length]
            if self.split == 'train':
                random.shuffle(indices)
            self.inputs.extend(list(chunks(self.len_dict[length], self.batch_size)))


    
    def load_seq_data(self, path):

        with open(path, 'r') as reader:
            # print(train_file_path)
            content = np.array([x.strip().split(', ') for x in reader.readlines()]).astype(float)
            downsample_content = content[0:content.shape[0]:self.downsample_rate]

        return downsample_content
    
    def __getitem__(self, index):
              
        
        indices = self.inputs[index]

        ret = {
            'input_x': [],

            'target_y': [],

            'action_lenth':[],
            
            'seq_info': []

        }
        
        for idx in indices:
            this_ret = self.getitem_one(idx)
            ret['input_x'].append(this_ret['input_x'])

            ret['target_y'].append(torch.as_tensor(this_ret['target_y']).type(torch.LongTensor))
            ret['action_lenth'].append(torch.as_tensor(this_ret['action_lenth']).type(torch.LongTensor))
            ret['seq_info'].append(this_ret['seq_info'])

            

        ret['input_x'] = torch.stack(ret['input_x'])

        ret['target_y'] = torch.stack(ret['target_y'])
        
        ret['action_lenth'] = torch.stack(ret['action_lenth'])
        # to locate image

        return ret
    
    def getitem_one(self, index):
        ret = {}
        
        sample_path = self.samples[index]
        label = self.labels[index]
        downsample_content = self.load_seq_data(sample_path)

        ret['input_x'] = torch.tensor(downsample_content, dtype=torch.float)

        ret['target_y'] = label
        ret['action_lenth'] = downsample_content.shape[0]
        ret['seq_info'] = sample_path

        return ret
    
    def __len__(self):
        return  len(self.inputs)   
    
def my_collate_fn(batch):
    return batch[0]   

def build_data_loader(args, phase='train', batch_size=None):
    data_loaders = data.DataLoader(
        dataset=DataLayer(args, phase),
        batch_size= 1,
        shuffle=phase=='train',
        num_workers= args.preprocess_workers,
        collate_fn=my_collate_fn if batch_size is not None else None)

    return data_loaders
if __name__ == '__main__':
    
    a = build_data_loader(args, 'train', batch_size = 1)

    for data in a:
        inputs = data
        g=1+1

