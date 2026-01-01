# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gaussian_kde
from scipy.integrate import trapezoid
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass
import warnings
from itertools import combinations

@dataclass
class SensitivityResults:
    """Container for sensitivity analysis results."""
    method: str
    first_order: Optional[Dict[str, float]] = None
    total_order: Optional[Dict[str, float]] = None
    second_order: Optional[Dict[Tuple[str,str],float]] = None
    local_sensitivity: Optional[Dict[str, np.ndarray]] = None
    parameter_names: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    
    def summary(self) -> str:
    """Generate summary string."""
    lines = [f"Sensitivity Analysis Results ({self.method})"] 
    lines.append("=" * 60)
    
    if self.first_order:
        lines.append("\nFirst-order indices (main effects):")
        for param, value in self.first_order.items():
            lines.append(f" {param:20s}: {value:8.4f}")
            
    if self.total_order:
        lines.append("\nTotal-order indices (main + interactions):")
        for param, value in self.total_order.items():
            lines.append(f" {param:20s}: {value:8.4f}")
                
    if self.second_order:
        lines.append("\nSecond-order indices (interactions):")
        for (p1, p2), value in list(self.second_order.items())[:10]:
            lines.append(f" {p1}-{p2:15s}: {value:8.4f}")       
    
    return "\n".join(lines)        

### Next Step: SensitivityAnalyzer class
