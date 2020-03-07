
import torch
from model import MultiStageModel
from train import Trainer
from predict import *
from batch_gen import BatchGenerator
import os
import argparse
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
# architecture
parser.add_argument('--num_stages', default=4, type=int, help='stage number')
parser.add_argument('--num_layers', default=10, type=int, help='layer number in each stage')
parser.add_argument('--num_f_maps', default=64, type=int, help='embedded feat. dim.')
parser.add_argument('--features_dim', default=2048, type=int, help='input feat. dim.')
parser.add_argument('--DA_adv', default='none', type=str, help='adversarial loss (none | rev_grad)')
parser.add_argument('--DA_adv_video', default='none', type=str, help='video-level adversarial loss (none | rev_grad | rev_grad_ssl | rev_grad_ssl_2)')
parser.add_argument('--pair_ssl', default='all', type=str, help='pair-feature methods for SSL-DA (all | adjacent)')
parser.add_argument('--num_seg', default=10, type=int, help='segment number for each video')
parser.add_argument('--place_adv', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_adv) == num_stages')
parser.add_argument('--multi_adv', default=['N', 'N'], type=str, nargs="+",
                    metavar='N', help='separate weights for domain discriminators')
parser.add_argument('--weighted_domain_loss', default='Y', type=str, help='weighted domain loss for class-wise domain discriminators')
parser.add_argument('--ps_lb', default='soft', type=str, help='pseudo-label type (soft | hard)')
parser.add_argument('--source_lb_weight', default='pseudo', type=str, help='label type for source data weighting (real | pseudo)')
parser.add_argument('--method_centroid', default='none', type=str, help='method to get centroids (none | prob_hard)')
parser.add_argument('--DA_sem', default='mse', type=str, help='metric for semantic loss (none | mse)')
parser.add_argument('--place_sem', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_sem) == num_stages')
parser.add_argument('--ratio_ma', default=0.7, type=float, help='ratio for moving average centroid method')
parser.add_argument('--DA_ent', default='none', type=str, help='entropy-related loss (none | target | attn)')
parser.add_argument('--place_ent', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ent) == num_stages')
parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'domain_attn'], help='attention mechanism')
parser.add_argument('--DA_dis', type=str, default='none', choices=['none', 'JAN'], help='discrepancy method for DA')
parser.add_argument('--place_dis', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_dis) == num_stages')
parser.add_argument('--DA_ens', type=str, default='none', choices=['none', 'MCD', 'SWD'], help='ensemble method for DA')
parser.add_argument('--place_ens', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ens) == num_stages')
parser.add_argument('--SS_video', type=str, default='none', choices=['none', 'VCOP'], help='video-based self-supervised learning method')
parser.add_argument('--place_ss', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ss) == num_stages')
# config & setting
parser.add_argument('--path_data', default='data/')
parser.add_argument('--path_model', default='models/')
parser.add_argument('--path_result', default='results/')
parser.add_argument('--action', default='train')
parser.add_argument('--use_target', default='none', choices=['none', 'uSv'])
parser.add_argument('--split_target', default='0', help='split for target data (0: no additional split for target)')
parser.add_argument('--ratio_source', default=1, type=float, help='percentage of total length to use for source data')
parser.add_argument('--ratio_label_source', default=1, type=float, help='percentage of labels to use for source data (after previous processing)')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
# hyper-parameters
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--bS', default=1, type=int, help='batch size')
parser.add_argument('--alpha', default=0.15, type=float, help='weighting for smoothing loss')
parser.add_argument('--tau', default=4, type=float, help='threshold to truncate smoothing loss')
parser.add_argument('--beta', default=[-2, -2], type=float, nargs="+", metavar='M', help='weighting for adversarial loss & ensemble loss ([frame-beta, video-beta])')
parser.add_argument('--iter_max_beta', default=[1000, 1000], type=float, nargs="+", metavar='M', help='for adaptive beta ([frame-beta, video-beta])')
parser.add_argument('--gamma', default=-2, type=float, help='weighting for semantic loss')
parser.add_argument('--iter_max_gamma', default=1000, type=float, help='for adaptive gamma')
parser.add_argument('--mu', default=1, type=float, help='weighting for entropy loss')
parser.add_argument('--nu', default=-2, type=float, help='weighting for the discrepancy loss')
parser.add_argument('--eta', default=1, type=float, help='weighting for the self-supervised loss')
parser.add_argument('--iter_max_nu', default=1000, type=float, metavar='M', help='for adaptive nu')
parser.add_argument('--dim_proj', default=128, type=int, help='projection dimension for SWD')
# runtime
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--verbose', default=False, action="store_true")
parser.add_argument('--use_best_model', type=str, default='none', choices=['none', 'source', 'target'], help='save best model')
parser.add_argument('--multi_gpu', default=False, action="store_true")
parser.add_argument('--resume_epoch', default=0, type=int)
# tensorboard
parser.add_argument('--use_tensorboard', default=False, action='store_true')
parser.add_argument('--epoch_embedding', default=50, type=int, help='select epoch # to save embedding (-1: all epochs)')
parser.add_argument('--stage_embedding', default=-1, type=int, help='select stage # to save embedding (-1: last stage)')
parser.add_argument('--num_frame_video_embedding', default=50, type=int, help='number of sample frames per video to store embedding')

args = parser.parse_args()

# check whether place_adv & place_sem are valid
if len(args.place_adv) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_sem) != args.num_stages:
    raise ValueError('len(place_sem) should be equal to num_stages')
if len(args.place_ent) != args.num_stages:
    raise ValueError('len(place_ent) should be equal to num_stages')
if len(args.place_dis) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_ens) != args.num_stages:
    raise ValueError('len(place_ens) should be equal to num_stages')
if len(args.place_ss) != args.num_stages:
    raise ValueError('len(place_ss) should be equal to num_stages')

if args.use_target == 'none':
    args.DA_adv = 'none'
    args.DA_sem = 'none'
    args.DA_ent = 'none'
    args.DA_dis = 'none'
    args.DA_ens = 'none'
    args.SS_video = 'none'  # focus on cross-domain setting

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# ====== Load files ====== #
vid_list_file = args.path_data+args.dataset+"/splits/train.split"+args.split+".bundle"

vid_list_file_target = args.path_data+args.dataset+"/splits/test.split"+args.split+".bundle"
vid_list_file_test = vid_list_file_target

if args.split_target != '0':
    vid_list_file_target = args.path_data + args.dataset + "/splits/test_train_" + args.split_target + ".split" + args.split + ".bundle"
    vid_list_file_test = args.path_data + args.dataset + "/splits/test_test_" + args.split_target + ".split" + args.split + ".bundle"

features_path = args.path_data+args.dataset+"/features/"
gt_path = args.path_data+args.dataset+"/groundTruth/"

mapping_file = args.path_data+args.dataset+"/mapping.txt"  # mapping between classes & indices

model_dir = args.path_model+args.dataset+"/split_"+args.split
results_dir = args.path_result+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]  # list of classes
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

# initialize model & trainer
model = MultiStageModel(args, num_classes)
trainer = Trainer(num_classes)

# ====== Main Program ====== #
start_time = time.time()
if args.action == "train":
    batch_gen_source = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_target = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_source.read_data(vid_list_file)  # read & shuffle the source training list
    batch_gen_target.read_data(vid_list_file_target)  # read & shuffle the target training list
    trainer.train(model, model_dir, results_dir, batch_gen_source, batch_gen_target, device, args)

if args.action == "predict":
    predict(model, model_dir, results_dir, features_path, vid_list_file_test, args.num_epochs, actions_dict,
            device, sample_rate, args)

end_time = time.time()

if args.verbose:
    print('')
    print('total running time:', end_time - start_time)
