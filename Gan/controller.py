
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import math

from IPython import display
import glob
import imageio
import os
from PIL import Image
import random
import copy 
from sys import argv 

import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#import tensorflow_gan as tfgan
IMG_SHAPE = (128, 128)
MAX_DATASET_SIZE = 10000