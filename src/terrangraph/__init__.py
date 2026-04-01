"""
TerranGraph - Graph-Based Geotechnical Site Representation
========================================================

TerranGraph is a Python framework for graph-based learning of geotechnical site
representation and subsurface modeling. It integrates multi-source geotechnical
data processing, spatial graph construction, graph neural networks, and 3D
visualization into a unified workflow for subsurface characterization and prediction.

Main Components:
- DataHandler: Data loading and preprocessing for geotechnical data
- GeoModel: Graph neural network modeling for site representation learning
- GeoVisualizer: 3D visualization using PyVista
- CLI: Command-line interface for workflow execution
- GUI: PyQt5-based interactive interface (optional)

Example:
    >>> from terrangraph import DataHandler, GeoModel, GeoVisualizer
    >>>
    >>> # Load and process data
    >>> handler = DataHandler()
    >>> handler.load_data("geotechnical_data.xlsx")
    >>>
    >>> # Generate spatial graph
    >>> coords, labels, edges = handler.generate_grid()
    >>>
    >>> # Train model
    >>> model = GeoModel()
    >>> model.prepare_data(coords, labels, edges)
    >>> model.build_model(num_classes=5)
    >>> model.train(epochs=100)
    >>>
    >>> # Visualize results
    >>> visualizer = GeoVisualizer()
    >>> predictions, _ = model.predict()
    >>> mesh = visualizer.create_model(coords, predictions)
    >>> visualizer.show_model(mesh)
"""

__version__ = "0.4.0"
__author__ = "Wang Lai"
__email__ = "wanglai@imust.edu.cn"
__license__ = "MIT"

# Core components
from .core.data_handler import DataHandler
from .core.geo_model import GeoModel, GNN
from .core.visualizer import GeoVisualizer

# Command line interface
from .cli import main as cli_main

# Public API
__all__ = [
    # Version info
    '__version__', '__author__', '__email__', '__license__',
    
    # Core classes
    'DataHandler', 'GeoModel', 'GNN', 'GeoVisualizer',
    
    # Utilities
    'validate_data_format', 'estimate_memory_usage', 'optimize_parameters',
    
    # CLI
    'cli_main',
    
    # GUI availability
    'GUI_AVAILABLE',
]
