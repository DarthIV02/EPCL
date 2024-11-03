#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.scene.visuals import Text
import numpy as np
from matplotlib import pyplot as plt
import time
from pcseg.model import build_network, load_data_to_gpu
import torch
import torch.nn as nn
import laspy
import os

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, 
               sem_color_dict,          #label color map
               semantics=True,          #if True, will print ground-truth data
               predictions=True,        #if True, will print predicted data. requires semantics=True
               verbose_runtime=False,   #if True, will print time taken to load/plot data
               csvwriter=None,          #writes timedata to external file
               pullData=None,           #callable for extracting data to plot
               percent_points = 1.0,
               inference_model = None,
               dataset=None,
               data_config = None,
               first = None):
    self.semantics = semantics
    self.predictions = predictions
    self.percent_points = percent_points
    self.pullData = pullData
    self.inference_model = inference_model
    self.dataset = dataset

    # useful for determining runtime data, not required
    self.csvwriter = csvwriter
    self.verbose_runtime = verbose_runtime

    # time (s) between each individual scan for timer-based visualization
    self.TIME_INTERVAL = 0.5

    # sanity check
    #if not self.semantics and self.instances:
    #  print("Instances are only allowed in when semantics=True")
    #  raise ValueError
    if not self.semantics and self.predictions:
      print("Predictions require ground truth visualization: semantics=True")
      raise ValueError
    
    self.sem_color_dict = sem_color_dict

    max_sem_key = 0
    for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
            max_sem_key = key + 1
    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    if dataset == 'tls':
      base = data_config['DATA_PATH'][:-33]
      self.lidars = os.listdir(base + 'lidar_test/train/')
      self.lidars = [base + 'lidar_test/train/' + i for i in self.lidars]
      self.lidars.sort()
      print(self.lidars)

    #self.real_i = 0
    print(first['name'])
    print(self.lidars[int(first['name'][0][-5])-1])
    las_file = laspy.read(self.lidars[int(first['name'][0][-5])-1])

    self.reset()
    load_data_to_gpu(first)
    with torch.no_grad():
        ret_dict = inference_model(first)
    pc, labels, pred = first['original_p'][0][:,:3].float(), ret_dict['point_labels'], ret_dict['point_predict']
    self.next_scan(pc, pred[0], labels[0], real=las_file)

  # method for clock event callback
  def next_scan(self, points, pred, labels, time_x=None, event=None, real=None):
    #print(points)
    #print(points.shape)
    # centroid = torch.mean(points, dim=0)
    points = points[:,:3].cpu().numpy()
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    data = self.prep_data((centered_points, pred, labels, time_x))
    if real != None:
      self.update_scan(data, real=real)
    else:
      self.update_scan(data)

  def reset(self):
    """ Reset. """
    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True, size=(1600, 600))
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    
    self.clock = vispy.app.timer.Timer(interval=self.TIME_INTERVAL, connect=self.next_scan)

    # grid
    self.grid = self.canvas.central_widget.add_grid()
    zoom = 10.0

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = vispy.scene.cameras.turntable.TurntableCamera(scale_factor = zoom)
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)    
    Text('Raw Scan', parent=self.scan_view, color='white', anchor_x="left", anchor_y="bottom", font_size=18)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = vispy.scene.cameras.turntable.TurntableCamera(scale_factor = zoom)
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      # synchronize raw and semantic cameras
      self.sem_view.camera.link(self.scan_view.camera)  
      Text('Ground Truth', parent=self.sem_view, color='white', anchor_x="left", anchor_y="bottom", font_size=18)  
    if self.predictions:
      print("Plotting predictions in visualizer")
      self.pred_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.pred_view, 0, 2)
      self.pred_vis = visuals.Markers()
      self.pred_view.camera = vispy.scene.cameras.turntable.TurntableCamera(scale_factor = zoom)
      self.pred_view.add(self.pred_vis)
      visuals.XYZAxis(parent=self.pred_view.scene)
      # synchronize raw and semantic cameras
      self.pred_view.camera.link(self.scan_view.camera)
      Text('Predicted Labels', parent=self.pred_view, color='white', anchor_x="left", anchor_y="bottom", font_size=18)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def get_colors(self, points, gt_labels, pred_labels, real=None):
        
        if len(pred_labels) == 1:
          pred_labels = pred_labels[0]
        if len(gt_labels) == 1:
          gt_labels = gt_labels[0]

        if self.dataset != 'tls':
          power = 16
          range_data = np.linalg.norm(points, 2, axis=1)
          range_data = range_data**(1 / power)
          viridis_range = ((range_data - range_data.min()) /
                          (range_data.max() - range_data.min()) *
                          255).astype(np.uint8)
          viridis_map = self.get_mpl_colormap("viridis")
          self.viridis_color = viridis_map[viridis_range]
        else:
          if max(real.red) > 255:
            red_8bit = (real.red / 256).astype(int).reshape(-1, 1) / 255.
            green_8bit = (real.green / 256).astype(int).reshape(-1, 1) / 255.
            blue_8bit = (real.blue / 256).astype(int).reshape(-1, 1) / 255.
          else:
            red_8bit = real.red.reshape(-1, 1) / 255.
            green_8bit = real.green.reshape(-1, 1) / 255.
            blue_8bit = real.blue.reshape(-1, 1) / 255.
          self.viridis_color = np.hstack((red_8bit, green_8bit, blue_8bit))
        
        sem_label = pred_labels #& 0xFFFF  # semantic label in lower half
        self.sem_label_color = self.sem_color_lut[sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        sem_label = gt_labels #& 0xFFFF  # semantic label in lower half
        self.sem_gt_label_color = self.sem_color_lut[sem_label]
        self.sem_gt_label_color = self.sem_gt_label_color.reshape((-1, 3))

  #@getTime
  def update_scan(self, data, real=None):
    points, pred_labels, gt_labels, t = data

    start = time.time()
    self.get_colors(points, gt_labels, pred_labels, real=real)
    load = time.time()

    # Raw scan -> change
    self.scan_vis.set_data(real.xyz,
                           face_color=self.viridis_color[..., ::-1],
                           edge_color=self.viridis_color[..., ::-1],
                           size=0.5)
    #print("Real Scan: ", self.lidars[int(first['name'][0][-5])])
    scan_data = time.time()
    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(points,
                            face_color=self.sem_gt_label_color[..., ::-1],
                            edge_color=self.sem_gt_label_color[..., ::-1],
                            size=1)
    sem_data = time.time()
    # plot predictions
    if self.predictions:
      self.pred_vis.set_data(points,
                             face_color=self.sem_label_color[..., ::-1],
                             edge_color=self.sem_label_color[..., ::-1],
                             size=1)
    pred_data = time.time()

    if self.verbose_runtime:
      print("Visualizing {0} points".format(len(points)))
      print("""Time Elapsed: 
            Loading Colors:\t\t{0}
            Plotting Raw:\t\t\t{1}
            Plotting Ground Truth:\t{2}
            Plotting Predictions:\t\t{3}
            Total time:\t\t\t{4}"""
            .format(load-start, scan_data-load, sem_data-scan_data, pred_data-sem_data, (pred_data-start) + t))
    
    '''if not(self.csvwriter == None):
      self.csvwriter.writerow({
        'Points':len(points),
        'LoadData':load-start,
        'PlotRaw':scan_data-load,
        'PlotSem':sem_data-scan_data})
    return'''

  def prep_data(self, data):
    if self.percent_points == 1:
      return data
    
    # generate bitmask
    mask = np.ones(len(data[0]), dtype=bool)
    mask[:int(len(mask)*(1-self.percent_points))] = False
    np.random.shuffle(mask)

    return data[0][mask, ...], data[1][mask, ...], data[2][mask, ...], data[3]

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    if event.key == 'N' and not self.clock.running:
      first = next(iter(self.pullData))
      load_data_to_gpu(first)
      with torch.no_grad():
          ret_dict = self.inference_model(first)
      pc, labels, pred = first['original_p'][0][:,:3].float(), ret_dict['point_labels'], ret_dict['point_predict']
      if self.dataset == 'tls':
        las_file = laspy.read(self.lidars[int(first['name'][0][-5])-1])
        self.next_scan(pc, pred, labels, real=las_file)
      else:
        self.next_scan(pc, pred, labels)
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()
    elif event.key == ' ':
      if not self.clock.running:
        self.clock.start()
      else:
        self.clock.stop()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
