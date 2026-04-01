# TerranGraph: A Graph-Based Framework for Geotechnical Site Representation

<p align="center">
  <img src="https://raw.githubusercontent.com/wanglai25/terrangraph/main/assets/banner-revised.png" alt="TerranGraph banner" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/terrangraph.svg)](https://pypi.org/project/terrangraph/)
[![Python version](https://img.shields.io/pypi/pyversions/terrangraph.svg)](https://pypi.org/project/terrangraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TerranGraph is a Python framework for geotechnical site representation, learning, and visualization. It integrates multi-source data processing, spatial graph construction, graph neural networks, and 3D visualization into a single workflow for subsurface modeling and prediction.

---

## Overview

TerranGraph follows an end-to-end workflow:

**Geotechnical Data → Spatial Graph → Representation Learning → Prediction → Visualization**

Designed for subsurface modeling and site characterization, TerranGraph bridges geotechnical data and graph-based intelligence.

---

## Features

- **Multi-source data integration** for Excel and CSV geotechnical datasets
- **Spatial graph construction** from borehole logs and spatial coordinates
- **Graph neural networks** for site representation learning and prediction
- **3D visualization** with PyVista for subsurface modeling and cross-section analysis
- **Interactive GUI** based on PyQt5 for workflow exploration
- **Scalable training** with PyTorch and optional GPU acceleration
- **Model export** to standard 3D formats such as PLY, OBJ, and STL

---

## Installation

```bash
pip install terrangraph
```

## Quick Start

```python
from terrangraph import DataHandler, GeoModel, GeoVisualizer

# Load geotechnical data
handler = DataHandler()
unique_types, num_classes = handler.load_data("your_dataset.xlsx")

# Preprocess and align coordinates
rect = handler.compute_mbr()
bounds = handler.align_coordinates(x_res=10, y_res=2, z_res=1)
params_dict = handler.create_param_dict()

# Construct spatial graph
coords, labels, edges = handler.generate_grid(bounds, r=2)

# Train graph neural network
model = GeoModel()
model.prepare_data(coords, labels, edges)
model.build_model(
    num_classes=num_classes,
    hidden_size=48,
    gcn_layers=3,
    mlp_layers=3,
    dropout=0.2
)
history = model.train(epochs=100, lr=3e-3)

# Prediction and visualization
predictions, probabilities = model.predict()
visualizer = GeoVisualizer()
mesh = visualizer.create_model(coords, predictions)
visualizer.show_model(mesh)
```

## Input Data Format

The input dataset should include:

- `X`, `Y`, `Z`: spatial coordinates
- `Soil/Rock Type`: soil or rock type classification
- `Borehole ID`: borehole identifier (optional)
- `Borehole Type`: borehole category (optional)

Additional geotechnical parameters (e.g., strength, density, CPT measurements, etc.) can be incorporated for extended modeling tasks.

## Use Cases

TerranGraph can be used for:

- 3D subsurface modeling
- Borehole-based site characterization
- Graph-based representation learning of geotechnical data
- Similarity-based site comparison and clustering
- Data-driven prediction of soil and rock properties

## Project Structure

```text
terrangraph/
├── src/
│   └── terrangraph/
│       ├── __init__.py              # Package exports
│       ├── cli.py                   # Command-line interface
│       └── core/
│           ├── __init__.py
│           ├── data_handler.py      # Data loading and preprocessing
│           ├── geo_model.py         # Graph neural network models
│           └── visualizer.py        # 3D visualization (PyVista)
│
├── assets/                          # Logo and banner images
├── docs/                            # Documentation files
├── CHANGELOG.md                     # Version history
├── LICENSE                          # License
├── pyproject.toml                   # Package configuration
└── README.md                        # Project documentation
```

## Examples

See the [examples/](examples/) directory:

- `basic_usage.py`: basic workflow
- `advanced_visualization.py`: advanced visualization
- `batch_processing.py`: batch processing
- `custom_model.py`: custom GNN architectures

## Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [User Tutorial](docs/tutorial.md)
- [Developer Guide](docs/development.md)

## System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (required for `ml`)
- OpenGL support for 3D visualization (required for `viz`)

## Roadmap

### Current Capabilities

- [x] Graph-based geotechnical site representation  
- [x] Spatial graph construction from borehole data  
- [x] GNN-based prediction framework  
- [x] 3D subsurface visualization (PyVista)  

### Next Steps

- [ ] Integration of additional geotechnical parameters (e.g., CPT, strength, density)
- [ ] Integration of real-time monitoring data (e.g., tunneling operation data)  
- [ ] Contrastive learning for site similarity analysis  
- [ ] Support for large-scale geotechnical datasets  
- [ ] Probabilistic modeling and uncertainty quantification  

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TerranGraph in your research, please cite:

```bibtex
@software{wang2026terrangraph,
  author = {Wang, Lai},
  title = {TerranGraph: A Graph-Based Framework for Geotechnical Site Representation},
  year = {2026},
  url = {https://github.com/wanglai25/terrangraph}
}
```

## Support

- Email: <wanglai@imust.edu.cn>
- Issues: [GitHub Issues](https://github.com/wanglai25/terrangraph/issues)
- Discussions: [GitHub Discussions](https://github.com/wanglai25/terrangraph/discussions)
