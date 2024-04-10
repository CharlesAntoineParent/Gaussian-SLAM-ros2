"""This modules implements a ros2 node interface for the Gaussian-SLAM system."""

import cv2
import message_filters
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from src.entities.gaussian_slam import GaussianSLAM
from src.utils.io_utils import load_config
from src.utils.vis_utils import *  # noqa - needed for debugging


class GaussianSplattingSlamNode(Node):
    """This nodes implements a ROS2 interface for the Gaussian-SLAM system."""
    
    def __init__(self, config_path: str):
        super().__init__('gaussian_splatting_slam_node')
        self.get_logger().info('Gaussian-SLAM node started')
        
        self.config = load_config(config_path)
        self.gslam = GaussianSLAM(self.config)
        self.gslam.init_scene()
        qos_profile = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.VOLATILE,
			history=HistoryPolicy.KEEP_LAST,
			depth=1
		)
        image_sub = message_filters.Subscriber(
            self, CompressedImage, "compressed_image", qos_profile=qos_profile
        )
        depth_sub = message_filters.Subscriber(
            self, Image, "depth", qos_profile=qos_profile
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub], queue_size=10, slop=2
        )
        self.ts.registerCallback(self.track_object)
        self.current_rgb_frame = None
        self.current_depth_frame = None

        
    def register_new_frame(self, image: CompressedImage, depth: Image) -> None:
        """Registers a new frame with the Gaussian-SLAM system.
        
            Args:
                image (CompressedImage): The compressed RGB image.
                depth (Image): The depth image.
            
        """
        cv_depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        cv_image = cv2.imdecode(np.asarray(image.data), cv2.IMREAD_COLOR)
        self.gslam.run_inference(cv_image, cv_depth)
        
        
        
        