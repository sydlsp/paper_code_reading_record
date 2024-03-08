from torch import nn
import torch
from torchvision import models
import torch.nn.functional as F
from modules import keypoint_detector
import math
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian



