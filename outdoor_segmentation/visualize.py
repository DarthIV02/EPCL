#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscanvis import LaserScanVis
from auxiliary.dataset import SemKITTI_sk
from torch.utils.data import DataLoader
from vispy.scene import SceneCanvas
from vispy.util.event import Event
import time
import importlib
import argparse
from omegaconf import OmegaConf
import os.path as osp
import torch 
from train import Trainer
from pcseg.model import build_network, load_data_to_gpu
import copy
from tools.utils.train.config import cfgs, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pathlib import Path

import csv
import datetime

def collate_fn_BEV(data):
    points = data[0][0]
    remissions = data[0][1]
    labels = data[0][2]
    viridis_colors = data[0][3]
    sem_label_colors = data[0][4]
    return points, remissions, labels, viridis_colors, sem_label_colors

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
    '-cfg', 
    '--cfg_file', 
    help='the path to the setup config file', 
    default='cfg/train_sk.yaml')
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds. '
      'Defaults to %(default)s',
  )
  parser.add_argument(
    '--log_path',
    dest='log_path',
    default=None,
    required=False,
    help='Path to log visualization time to .csv file. Requires log_data argument. '
    'Defaults to %(default)s',
  )
  parser.add_argument(
    '--log_data',
    dest='log_data',
    default=False,
    action='store_true',
    help='Records and stores runtime for scanning and plotting visualizaions. Defaults to False'
  )
  parser.add_argument(
    '--print_data',
    dest='print_data',
    default=False,
    action='store_true',
    help='Enables printing of runtime data to terminal. Defaults to False.'
  )
  parser.add_argument(
    '--enable_auto',
    dest='enable_auto',
    default=False,
    required=False,
    action='store_true',
    help='Enables instantaneous visualization of files for collecting'
    ' large amounts of visualization time measurements',
  )
  parser.add_argument(
    '--shuffle',
    dest='shuffle',
    default=False,
    required=False,
    action='store_true',
    help='Shuffles scans before visualization. Defaults to False'
  )
  # Trainer arguments
  parser.add_argument(
    '--eval',
    dest='eval',
    default=True,
    required=False,
    help='eval flag',
  )
  parser.add_argument(
    '--exp',
    dest='exp',
    default=5,
    required=False,
    help='exp flag',
  )
  parser.add_argument(
    '--pretrained_model',
    dest='pretrained_model',
    default='/root/main/EPCL_setup/checkpoints/best_checkpoint.pth',
    help='pretrained model path',
  )
  parser.add_argument(
    '--launcher',
    dest='launcher',
    default='pytorch',
    help='launcher',
  )
  parser.add_argument(
    '--extra_tag',
    dest='extra_tag',
    default='val_EPCL_HD_tls',
    required=False,
    help='eval flag',
  )
  parser.add_argument('--tcp_port', type=int, default=18888,
                        help='tcp port for distrbuted training')
  parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
  parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
  parser.add_argument('--batch_size', type=int, default=None, required=False,
                        help='batch size for model training.')
  parser.add_argument('--fix_random_seed', action='store_true', default=True,
                        help='whether to fix random seed.')
  parser.add_argument('--crop', action='store_true', default=False,
                        help='Choose encoding')
  parser.add_argument('--epochs', type=int, default=None, required=False,
                        help='number of epochs for model training.')
  parser.add_argument('--sync_bn', action='store_true', default=False,
                      help='whether to use sync bn.')
  parser.add_argument('--ckp', type=str, default=None,
                      help='checkpoint to start from')
  parser.add_argument('--amp', action='store_true', default=False,
                        help='whether to use mixture precision training.')
  parser.add_argument('--ckp_save_interval', type=int, default=1,
                      help='number of training epochs')
  parser.add_argument('--max_ckp_save_num', type=int, default=30,
                      help='max number of saved checkpoint')
  parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False,
                      help='')
  # == hd configs ==
  parser.add_argument('--lr', type=float, default=0.01, required=False,
                      help='Learning rate for HD.')
  parser.add_argument('--train_hd', action='store_true', default=False,
                        help='only perform training on hd')
  parser.add_argument('--eval_interval', type=int, default=50,
                      help='number of training epochs')
  # == device configs ==
  parser.add_argument('--workers', type=int, default=1,  
                      help='number of workers for dataloader') 
  FLAGS, unparsed = parser.parse_known_args()
  cfg_from_yaml_file(FLAGS.cfg_file, cfgs)
  cfgs.TAG = Path(FLAGS.cfg_file).stem
  cfgs.EXP_GROUP_PATH = '/'.join(FLAGS.cfg_file.split('/')[2:-1])

  if FLAGS.set_cfgs is not None:
      cfg_from_list(FLAGS.set_cfgs, cfgs)

  #Get info relative to the set
  trainer = Trainer(FLAGS, cfgs)

  trainer.cur_epoch -= 1
  trainer.model.eval()
  data_config = copy.deepcopy(cfgs.DATA)
  from pcseg.data import build_dataloader

  _, test_loader, _ = build_dataloader(
      data_cfgs=data_config,
      modality=cfgs.MODALITY,
      batch_size=1,#cfgs.OPTIM.BATCH_SIZE_PER_GPU,
      dist=trainer.if_dist_train,
      workers=FLAGS.workers,
      logger=trainer.logger,
      training=False,
  )

  with open(f"./pcseg/data/dataset/tls/tls.yaml") as stream:
    try:
        colors = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
  
  color_dict = colors['color_map']

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Config", FLAGS.cfg_file)
  print("Sequence", FLAGS.sequence)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("log_path", FLAGS.log_path)
  print("log_data", FLAGS.log_data)
  print("print_data", FLAGS.print_data)
  print("enable_auto", FLAGS.enable_auto)
  print("shuffle", FLAGS.shuffle)
  print("*" * 80)

  # prevent updating log_path if log_data not used
  if FLAGS.log_path and not FLAGS.log_data:
    print("Must pass log_data argument to specify log_path")
    quit()

  # Require log_path argument if log_data is passed
  if FLAGS.log_data and not FLAGS.log_path:
    print("Must specify a log_path")
    quit()

  def temp(i):
    start = time.time()
    this = next(test_loader)
    load_data_to_gpu(this)
    with torch.no_grad():
        ret_dict = trainer.model(this)
    points, labels, predict = this['lidar'].C.float(), ret_dict['point_labels'], ret_dict['point_predict']
    end = time.time()
    print("Loaded points in {0} seconds".format(end-start))
    return points, labels, labels, end-start

  # create a visualizer
  # TODO update class variables
  vis = LaserScanVis( color_dict,
                      semantics=(not FLAGS.ignore_semantics),
                      verbose_runtime=FLAGS.print_data, 
                      pullData=temp,
                      percent_points=1,    
                      inference_model=trainer.model,
                      first = next(test_loader))
                    #key_press=key_press,
                    #canvas = canvas)

  # print instructions
  print("To navigate:")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")
  
  # if log_data flag is false, do not specify filewriter
  if not FLAGS.log_data:
    # run visualizer
    vis.run()

    quit()
