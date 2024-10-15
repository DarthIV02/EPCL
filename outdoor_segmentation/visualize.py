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
from datasets.inference_dataset import *
from datasets import *
import torch 

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
    '-cfg', 
    '--config', 
    help='the path to the setup config file', 
    default='cfg/train_sk.yaml')
  FLAGS, unparsed = parser.parse_known_args()

  cfg = OmegaConf.load(FLAGS.config)
  cluster_cfg = OmegaConf.load(cfg.cluster_cfg)
  model_cfg = OmegaConf.load(cfg.model_cfg)
  cfg = OmegaConf.merge(cfg,cluster_cfg,model_cfg)

  #Get info relative to the set
  if cfg.source == "semantickitti":
      source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
      train_set = SemanticKITTI(source_data_cfg,'train')
  elif cfg.source == "nuscenes":
      source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
      train_set = nuScenes(source_data_cfg,'train')
  else:
      raise  NameError('source dataset not supported')

  if cfg.target == "semantickitti":
      target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
      train_set_2 = SemanticKITTI(target_data_cfg,'train')
      target_set = SemanticKITTI(target_data_cfg,'valid')
  elif cfg.target == "nuscenes":
      target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes_mini.yaml")) # Change if using full nuscenes
      train_set_2 = nuScenes(target_data_cfg,'train')
      target_set = nuScenes(target_data_cfg,'valid')
  elif cfg.target == "semanticposs":
      target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semanticposs.yaml"))
      train_set_2 = SemanticPOSS(target_data_cfg,'train')
      target_set = SemanticPOSS(target_data_cfg,'valid')
  elif cfg.target == "semantickitti-nuscenes":
      target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti-nuscenes.yaml"))
      train_set_2 = SemanticKITTI_Nuscenes(target_data_cfg,'train')
      target_set = SemanticKITTI_Nuscenes(target_data_cfg,'valid')
  elif "pandaset" in cfg.target:
      target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,cfg.target+".yaml"))
      train_set_2 = Pandaset(target_data_cfg,'train')
      target_set = Pandaset(target_data_cfg,'valid')
  
  else:
      raise  NameError('target dataset not supported')
  
  color_dict = target_set.color_map

  #Get info relative to the model
  if cfg.architecture.model == "KPCONV":
      module = importlib.import_module('models.kpconv.kpconv')
      model_information = getattr(module, cfg.architecture.type)()
      model_information.num_classes = train_set.get_n_label()
      model_information.ignore_label = -1
      model_information.in_features_dim = model_cfg.architecture.n_features
      model_information.train_hd = cfg.train_hd
      from models.kpconv_model import SemanticSegmentationModel
      module = importlib.import_module('models.kpconv.architecture')
      model_type = getattr(module, cfg.architecture.type)
      model = SemanticSegmentationModel(model_information,cfg,model_type)
  elif cfg.architecture.model == "SPVCNN":
      module = importlib.import_module('models.spvcnn.spvcnn')
      model_information = getattr(module, cfg.architecture.type)
      model_information.num_classes = train_set.get_n_label()
      model_information.ignore_label = -1
      model_information.in_features_dim = model_cfg.architecture.n_features
      from models.spvcnn_model import SemanticSegmentationSPVCNNModel
      model = SemanticSegmentationSPVCNNModel(model_information,cfg)
  else:
      raise  NameError('model not supported')
      
  # Get HD info
  if cfg.train_hd or cfg.test_hd:
      hd_cfg = OmegaConf.load(cfg.hd_param)
      cfg = OmegaConf.merge(cfg,hd_cfg) 
      from models.HD import OnlineHD
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      #device = torch.device("cpu")
      model_hd = OnlineHD(hd_cfg.n_features, hd_cfg.n_dimensions, hd_cfg.n_classes, epochs = hd_cfg.epochs, device=device)
  
  # Define the path for the "HD" folder
  hd_folder = os.path.join(cfg.save_pred_path, f'HD_{hd_cfg.hd_block_stop}')

  # Check if the "HD" folder exists
  if not os.path.exists(hd_folder):
      os.makedirs(hd_folder)
      print(f"Folder 'HD' created at {hd_folder}")
  else:
      print(f"Folder 'HD' already exists at {hd_folder}")
  
  output_dataset = InferenceDataset(cfg, train_set, target_set, model, model_information, model_hd)

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
    points, labels = output_dataset.compute_sequence(0,i)
    end = time.time()
    print("Loaded points in {0} seconds".format(end-start))
    return points, labels, labels, end-start

  # create a visualizer
  # TODO update class variables
  vis = LaserScanVis(color_dict,
                      semantics=(not FLAGS.ignore_semantics),
                      verbose_runtime=FLAGS.print_data, 
                      pullData=temp,
                      percent_points=1,    
                      inference_model=output_dataset)
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
