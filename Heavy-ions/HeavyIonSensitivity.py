# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gaussian_kde
from scipy.integrate import trapezoid
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass
import warnings
from itertools import combinations
