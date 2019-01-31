#!/usr/bin/env python

from test_base import *
import sklearn.metrics
import numpy as np
from DLoader import *

class np_loader(DLoader):
	def __init__(self, db):
		
		DLoader.__init__(self, db)

