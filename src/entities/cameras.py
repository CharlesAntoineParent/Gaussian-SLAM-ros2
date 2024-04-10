import math

import numpy as np


class BaseCamera:

    def __init__(self, dataset_config: dict):
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])