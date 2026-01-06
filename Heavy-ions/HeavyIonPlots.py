#!/usr/bin/env python3

"""
This script generates plots / figures for Bayesian analysis results in heavy-ion collisions
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from HeavyIonMCMC import Chain
from HeavyIonEmulators import predictions 
