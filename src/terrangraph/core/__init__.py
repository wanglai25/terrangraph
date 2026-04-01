"""
Core modules for TerranGraph
==========================

This package contains the core functionality of TerranGraph:

- data_handler: Data loading and preprocessing for geotechnical data
- geo_model: Graph neural network modeling for site representation and prediction
- visualizer: 3D visualization with PyVista
"""

from .data_handler import DataHandler
from .geo_model import GeoModel, GNN
from .visualizer import GeoVisualizer

__all__ = [
    'DEFAULT_MODEL_PARAMS', 'SUPPORTED_FORMATS', 'setup_logging', 'set_seed', 'configure_plots',
    'DataHandler', 'GeoModel', 'GNN', 'GeoVisualizer'
]