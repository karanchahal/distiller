import os
import sys


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# avoid annoying import errors...
sys.path.append(FILE_DIR)

import models.cifar10 as cifar10
import models.cifar10sm as cifar10sm
import models.vision as vision
import models.wide_resnet as wide
