# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
import warnings
from dataclasses import dataclass
