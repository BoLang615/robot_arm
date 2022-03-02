import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)



# Data Parameters

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')
parser.add_argument('--dataroot',
                    help='dataset path',
                    type=str,
                    default='./dataset')


# Training Parameters


parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)


parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--downsample_rate',
                    help='downsample rate for each episode',
                    type=int,
                    default=10)

parser.add_argument('--sliding_window',
                    help='size for each sliding window',
                    type=int,
                    default=20)

parser.add_argument("--epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=10)

parser.add_argument('--lr',
                    help='learning rate',
                    type=float,
                    default=1e-3)

parser.add_argument('--classes',
                    help='num of action classes',
                    type=int,
                    default=3)

parser.add_argument('--checkpoint', default='', type=str)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--save_prediction',
                    help='save inference result to a txt file',
                    action='store_true')
args = parser.parse_args()