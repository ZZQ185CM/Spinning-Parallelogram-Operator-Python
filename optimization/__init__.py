"""
Optimization module for SPO algorithm
Contains guided filter, box filter, and other optimization utilities
"""

from .boxfilter import boxfilter
from .guidedfilter_color import guidedfilter_color
from .quantiz import quantiz
from .costagg import cost_agg
from .weighted_median_filter import weighted_median_filter
from .graphcuts import graph_cuts, graph_cuts_full

__all__ = [
    'boxfilter', 
    'guidedfilter_color', 
    'quantiz', 
    'cost_agg', 
    'weighted_median_filter',
    'graph_cuts',
    'graph_cuts_full'
]
