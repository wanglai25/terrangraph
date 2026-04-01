# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- Planned improvements and new features under development

---

## [0.3.0] - 2026-04-01

### Added

- Added Excel data support dependencies (`openpyxl`, `xlrd`) for improved compatibility with `.xlsx` and `.xls` formats

### Improved

- Enhanced data loading robustness for geotechnical datasets
- Improved compatibility of `DataHandler` with Excel-based workflows
- Reduced failure cases in data preprocessing and pipeline initialization

### Fixed

- Fixed errors caused by missing Excel dependencies during data loading
- Prevented cascading failures when data loading fails (e.g., MBR, grid generation, model initialization)

---

## [0.2.0] - 2026-04-01

### Changed

- Improved README layout with refined banner, badges, and visual hierarchy
- Updated project metadata and packaging configuration (`pyproject.toml`)

### Improved

- Improved project branding and presentation (banner design and layout)

---

## [0.1.0] - 2026-04-01

### Added

- Initial release of **TerranGraph**
- A unified graph-based framework for geotechnical site representation and learning
- Spatial graph construction from borehole and subsurface data
- Graph Neural Network (GNN) models for site representation and prediction
- 3D visualization of subsurface characteristics using PyVista
- Cross-section analysis tools for geotechnical interpretation
- Command-line interface (CLI) for training, prediction, and visualization
- Support for Excel and CSV geotechnical datasets
- Model export to standard 3D formats (PLY, OBJ, STL)
- Modular architecture for extensibility across data processing, modeling, and visualization
- Example workflows and user documentation
