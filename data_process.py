import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle

#

history_frames = 20 # 3 second * 2 frame/second


downsample_rate = 10
# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
# total_feature_dimension = 10 + 1 # we add mark "1" to the end of each row to indicate that this row exists
total_feature_dimension = 5 + 1

def generate_test_data(pra_file_path, seq_id):

    with open(pra_file_path, 'r') as reader:
        # print(train_file_path)
        content = np.array([x.strip().split(', ') for x in reader.readlines()]).astype(float)
        downsample_content = content[0:content.shape[0]:downsample_rate]
    frame_id_set = np.arange(len(downsample_content)) #sorted(set(now_dict.keys())) 

    all_feature_list = []

    # get all start frame id
    start_frame_id_list = frame_id_set[::history_frames]
    for start_ind in start_frame_id_list:
        start_ind = int(start_ind)
        end_ind = int(start_ind + history_frames)

        # print(start_ind, end_ind)
        now_frame_feature = np.array(content[start_ind:end_ind])
        if seq_id is not None:
            seq_id_array = np.ones((now_frame_feature.shape[0], 1 )) * seq_id
            now_frame_feature = np.concatenate([seq_id_array, now_frame_feature], axis=1)

        all_feature_list.append(now_frame_feature)


    # (N, T, C) 
    all_feature_list = np.array(all_feature_list)

    return all_feature_list


def generate_data(pra_file_path_list, save_path='train_data.pkl'):
    all_data = []

    for file_path in pra_file_path_list:
        seq_id = int(file_path.rstrip('.txt').split('/')[-1])
        now_data = generate_test_data(file_path, seq_id)

        all_data.extend(now_data)


    all_data = np.array(all_data) 


    print(np.shape(all_data))


    with open(save_path, 'wb') as writer:
        pickle.dump(all_data, writer)


if __name__ == '__main__':

    

	test_file_path_list = sorted(glob.glob(os.path.join('dataset/test/drawerpickstir40', '*.txt')))   



	
	print('Generating Testing Data.')
	generate_data(test_file_path_list, save_path='test_data.pkl')