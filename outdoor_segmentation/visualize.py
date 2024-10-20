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
    '--extra_tag',
    dest='extra_tag',
    default='val_EPCL_HD',
    required=False,
    help='eval flag',
  )
  FLAGS, unparsed = parser.parse_known_args()

  #Get info relative to the set
  trainer = Trainer(FLAGS, unparsed)

  trainer.cur_epoch -= 1
  trainer.model.eval()
  data_config = copy.deepcopy(unparsed.DATA)
  from pcseg.data import build_dataloader

  _, test_loader, _ = build_dataloader(
      data_cfgs=data_config,
      modality=unparsed.MODALITY,
      batch_size=1,#cfgs.OPTIM.BATCH_SIZE_PER_GPU,
      dist=trainer.if_dist_train,
      workers=FLAGS.workers,
      logger=trainer.logger,
      training=False,
  )
  
  color_dict = data_config.color_map

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Config", FLAGS.config)
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
                      inference_model=trainer.model.forward(),
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
