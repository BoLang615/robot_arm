
import os
import sys
from unittest import TestLoader

import numpy as np

import glob
from torch.utils import data
from argument_parser import args
import json
import pickle
import random
import torch

from torch.utils import data
# import debugpy
# debugpy.listen(5679)
# debugpy.wait_for_client()


    


def data_loader(args, pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False):
    testloader = TestLoader(args, data_path=pra_path)
    loader = data.DataLoader(
        dataset=testloader,
        batch_size=pra_batch_size,
        shuffle=pra_shuffle,
        drop_last=pra_drop_last, 
        num_workers=0,
        )
    return loader


class TestLoader(data.Dataset):
    """ testloader for action recognition
    Arguments:
        data_path: the path to '.pkl' data, the shape of data should be (N, T, C)
    """

    def __init__(self, args, data_path):
        '''
        train_val_test: (train, val, test)
        '''
        self.data_path = data_path
        self.downsample_rate = args.downsample_rate
        self.sliding_window = args.sliding_window
        test_file_path_list = sorted(glob.glob(os.path.join(self.data_path, '*.txt')))
        print('Generating Testing Data.')
        all_feature = []

        for file_path in test_file_path_list:
            seq_id = int(file_path.rstrip('.txt').split('/')[-1])
            now_data = self.generate_test_data(file_path, seq_id)

            all_feature.extend(now_data)

        self.all_feature = np.array(all_feature) 



    def __len__(self):
        return len(self.all_feature)

    def __getitem__(self, idx):
        # C = 18: 17 + [label]
        ret = {}
        now_feature = self.all_feature[idx].copy() # (T, C)
        ret['input_x'] = torch.from_numpy(now_feature[:,1:-1]).type(torch.float)
        if np.all(np.diff(now_feature[:,-1]) == 0):
            ret['target_y'] = torch.as_tensor(int(now_feature[0,-1])).type(torch.LongTensor)
            ret['split_index'] = np.array([20])
        else:
            split_index = np.where(np.diff(now_feature[:,-1]) != 0)[0]
            ret['split_index'] = split_index
            ret['target_y'] = torch.from_numpy(now_feature[:,-1]).type(torch.LongTensor)
        ## get seq info ###
        assert np.all(np.diff(now_feature[:,0:1]) == 0)
        seq_id = int(now_feature[0,0])
        ret['seq_info'] = os.path.join(self.data_path, '{:03d}.txt').format(seq_id)

        
        return ret

    def generate_test_data(self, pra_file_path, seq_id):

        with open(pra_file_path, 'r') as reader:
            # print(train_file_path)
            content = np.array([x.strip().split(', ') for x in reader.readlines()]).astype(float)
            downsample_content = content[0:content.shape[0]:self.downsample_rate]
        frame_id_set = np.arange(len(downsample_content)) #sorted(set(now_dict.keys())) 

        all_feature_list = []

        # get all start frame id
        start_frame_id_list = frame_id_set[:-self.sliding_window+1]
        for start_ind in start_frame_id_list:
            start_ind = int(start_ind)
            end_ind = int(start_ind + self.sliding_window)
            if end_ind > len(downsample_content):
                continue
            # print(start_ind, end_ind)
            now_frame_feature = np.array(downsample_content[start_ind:end_ind])
            if seq_id is not None:
                seq_id_array = np.ones((now_frame_feature.shape[0], 1 )) * seq_id
                now_frame_feature = np.concatenate([seq_id_array, now_frame_feature], axis=1)

            all_feature_list.append(now_frame_feature)


        # (N, T, C) 
        all_feature_list = np.stack(all_feature_list)

        return all_feature_list



   

if __name__ == '__main__':
    test_root = './dataset/test/drawerpickstir40'
    a = data_loader(args, test_root, pra_batch_size=1)

    for data in a:
        inputs = data
        g=1+1

