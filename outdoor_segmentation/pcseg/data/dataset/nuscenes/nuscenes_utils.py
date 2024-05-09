LABEL_NAME_MAPPING = {
    0: 'noise',
    1: 'animal',
    2: 'human.pedestrian.adult',
    3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker',
    5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer',
    7: 'human.pedestrian.stroller',
    8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier',
    10: 'movable_object.debris',
    11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone',
    13: 'static_object.bicycle_rack',
    14: 'vehicle.bicycle',
    15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid',
    17: 'vehicle.car',
    18: 'vehicle.construction',
    19: 'vehicle.emergency.ambulance',
    20: 'vehicle.emergency.police',
    21: 'vehicle.motorcycle',
    22: 'vehicle.trailer',
    23: 'vehicle.truck',
    24: 'flat.driveable_surface',
    25: 'flat.other',
    26: 'flat.sidewalk',
    27: 'flat.terrain',
    28: 'static.manmade',
    29: 'static.other',
    30: 'static.vegetation',
    31: 'vehicle.ego'}

color_map = {
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  2: [245, 150, 100],
  3: [245, 230, 100],
  4: [250, 80, 100],
  5: [150, 60, 30],
  6: [255, 0, 0],
  7: [180, 30, 80],
  8: [255, 0, 0],
  9: [30, 30, 255],
  10: [200, 40, 255],
  11: [90, 30, 150],
  12: [255, 0, 255],
  13: [255, 150, 255],
  14: [75, 0, 75],
  15: [75, 0, 175],
  16: [0, 200, 255],
  17: [50, 120, 255],
  18: [0, 150, 255],
  19: [170, 255, 150],
  20: [0, 175, 0],
  21: [0, 60, 135],
  22: [80, 240, 150],
  23: [150, 240, 255],
  24: [0, 0, 255],
  25: [255, 255, 50],
  26: [245, 150, 100],
  27: [255, 0, 0],
  28: [200, 40, 255],
  29: [30, 30, 255],
  30: [90, 30, 150],
  31: [250, 80, 100],
}

LEARNING_MAP = {
    0: 0,  # "unlabeled"
    1: 1,  # "outlier" mapped to "unlabeled" --------------------------mapped
    2: 2,  # "car"
    3: 3,  # "bicycle"
    4: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    5: 5,  # "motorcycle"
    6: 6,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    7: 7,  # "truck"
    8: 8,  # "other-vehicle"
    9: 9,  # "person"
    10: 10,  # "bicyclist"
    11: 11,  # "motorcyclist"
    12: 12,  # "road"
    13: 13,  # "parking"
    14: 14,  # "sidewalk"
    15: 15,  # "other-ground"
    16: 16,  # "building"
    17: 17,  # "fence"
    18: 18,  # "other-structure" mapped to "unlabeled" ------------------mapped
    19: 19,  # "lane-marking" to "road" ---------------------------------mapped
    20: 20,  # "vegetation"
    21: 21,  # "trunk"
    22: 22,  # "terrain"
    23: 23,  # "pole"
    24: 24,  # "traffic-sign"
    25: 25,  # "other-object" to "unlabeled" ----------------------------mapped
    26: 26,  # "moving-car" to "car" ------------------------------------mapped
    27: 27,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    28: 28,  # "moving-person" to "person" ------------------------------mapped
    29: 29,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    30: 30,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    31: 31,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
}

LEARNING_MAP_INV = {  # inverse of previous map
    0: 0,  # "unlabeled"
    1: 1,  # "outlier" mapped to "unlabeled" --------------------------mapped
    2: 2,  # "car"
    3: 3,  # "bicycle"
    4: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    5: 5,  # "motorcycle"
    6: 6,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    7: 7,  # "truck"
    8: 8,  # "other-vehicle"
    9: 9,  # "person"
    10: 10,  # "bicyclist"
    11: 11,  # "motorcyclist"
    12: 12,  # "road"
    13: 13,  # "parking"
    14: 14,  # "sidewalk"
    15: 15,  # "other-ground"
    16: 16,  # "building"
    17: 17,  # "fence"
    18: 18,  # "other-structure" mapped to "unlabeled" ------------------mapped
    19: 19,  # "lane-marking" to "road" ---------------------------------mapped
    20: 20,  # "vegetation"
    21: 21,  # "trunk"
    22: 22,  # "terrain"
    23: 23,  # "pole"
    24: 24,  # "traffic-sign"
    25: 25,  # "other-object" to "unlabeled" ----------------------------mapped
    26: 26,  # "moving-car" to "car" ------------------------------------mapped
    27: 27,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    28: 28,  # "moving-person" to "person" ------------------------------mapped
    29: 29,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    30: 30,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    31: 31,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
}
LEARNING_IGNORE = {  # Ignore classes
    0: True,  # "unlabeled", and others ignored
    1: False,  # "car"
    2: False,  # "bicycle"
    3: False,  # "motorcycle"
    4: False,  # "truck"
    5: False,  # "other-vehicle"
    6: False,  # "person"
    7: False,  # "bicyclist"
    8: False,  # "motorcyclist"
    9: False,  # "road"
    10: False,  # "parking"
    11: False,  # "sidewalk"
    12: False,  # "other-ground"
    13: False,  # "building"
    14: False,  # "fence"
    15: False,  # "vegetation"
    16: False,  # "trunk"
    17: False,  # "terrain"
    18: False,  # "pole"
    19: False,  # "traffic-sign"
}
