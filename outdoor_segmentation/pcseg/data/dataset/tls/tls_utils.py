LABEL_NAME_MAPPING = {
    0: 'unlabeled',
    1: 'alive',
    2: '1h',
    3: '10h',
    4: '100h',
    5: '1000h',}

color_map = {
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  2: [245, 150, 100],
  3: [245, 230, 100],
  4: [250, 80, 100],
  5: [150, 60, 30],
}

LEARNING_MAP = {
    0: 0, #noise
    1: 1, #noise
    2: 2, #noise
    3: 3, #noise
    4: 4, #noise
    5: 5, #noise
}

LEARNING_MAP_INV = {  # inverse of previous map
    0: 0,  # "unlabeled"
    1: 1,  # "outlier" mapped to "unlabeled" --------------------------mapped
    2: 2,  # "car"
    3: 3,  # "bicycle"
    4: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    5: 5,  # "motorcycle
}
LEARNING_IGNORE = {  # Ignore classes
    0: True,  # "unlabeled", and others ignored
    1: False,  # "car"
    2: False,  # "bicycle"
    3: False,  # "motorcycle"
    4: False,  # "truck"
    5: False,  # "other-vehicle"
}
