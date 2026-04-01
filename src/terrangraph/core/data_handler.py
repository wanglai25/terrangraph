"""
TerranGraph - Data Handler Module
==============================

Data handling module for TerranGraph.

This module provides functionalities for:
- Multi-source geotechnical data loading and preprocessing
- Spatial coordinate transformation and normalization
- Grid generation for subsurface representation
- Construction of graph structures for learning tasks
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

class DataHandler:
    """
    Data handler for geotechnical datasets.

    This class supports data loading, preprocessing, spatial alignment, and graph-ready grid generation for site representation learning.
    """
    def __init__(self):
        self.data = None
        self.soil_map = None
        self.original_coords = None
        self.mbr = None
        self.params_dict = {}
        self.logger = logging.getLogger(__name__ + ".DataHandler")
    
    def load_data(self, file_path):
        """
        Load geotechnical data from file (Excel, CSV)

        Parameters:
        -----------
        file_path : str
            Path to the data file

        Returns:
        --------
        tuple
            (soil_types: list, n_types: int)
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.xlsx' or file_ext == '.xls':
                self.data = pd.read_excel(file_path).ffill()
            elif file_ext == '.csv':
                self.data = pd.read_csv(file_path).ffill()
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return False, [], 0
            
            # Map soil types to numeric values
            unique_types = list(self.data['Soil type'].dropna().unique())
            self.soil_map = {v: i for i, v in enumerate(unique_types)}
            self.reverse_soil_map = {i: v for v, i in self.soil_map.items()}
            self.data['Soil type'] = self.data['Soil type'].map(self.soil_map)
            
            self.logger.info(f"Data loaded successfully from {file_path}")
            self.logger.info(f"Found {len(self.soil_map)} unique soil types: {unique_types}")
            return unique_types, len(unique_types)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return [], 0
    
    def compute_mbr(self):
        """
        Compute minimum bounding rectangle (MBR) of spatial points.
        
        Returns:
        --------
        np.ndarray
            Vertices of the minimum bounding rectangle
        """
        try:
            # Get unique locations
            locs = self.data[['X', 'Y']].drop_duplicates().values
            
            # Create convex hull
            hull = ConvexHull(locs)
            hull_pts = locs[hull.vertices]
            
            # Find minimum area oriented bounding box
            angles = np.unique(np.abs(np.arctan2(
                np.diff(hull_pts, axis=0)[:, 1], 
                np.diff(hull_pts, axis=0)[:, 0]) % (np.pi / 2)))
            
            rots = np.array([
                [[np.cos(a), np.cos(a - np.pi/2)], 
                 [np.sin(a), np.sin(a - np.pi/2)]] 
                for a in angles])
            
            proj = np.einsum('aij,jk->aik', rots, hull_pts.T)
            mins, maxs = proj.min(2), proj.max(2)
            areas = np.prod(maxs - mins, axis=1)
            best_idx = np.argmin(areas)
            best_rot = rots[best_idx]
            
            rect = np.dot([
                [mins[best_idx, 0], mins[best_idx, 1]],
                [maxs[best_idx, 0], mins[best_idx, 1]],
                [maxs[best_idx, 0], maxs[best_idx, 1]],
                [mins[best_idx, 0], maxs[best_idx, 1]]
            ], best_rot.T)
            
            self.mbr = rect
            self.logger.info("Minimum bounding rectangle computed")
            
            return rect
        
        except Exception as e:
            self.logger.error(f"Error computing MBR: {str(e)}")
            return None
    
    def align_coordinates(self, x_res, y_res, z_res):
        """
        Align coordinates to a grid based on the MBR.

        Returns
        -------
        dict
            bounds = {
                'x_min': int, 'x_max': int,
                'y_min': int, 'y_max': int,
                'z_min': int, 'z_max': int
            }
        """
        try:
            if self.mbr is None:
                self.compute_mbr()
            A, B, C, D = self.mbr
            origin = A
            U, V = B - A, D - A
            Uu = U / np.linalg.norm(U)
            Vu = V / np.linalg.norm(V)
        
            self.data[['X', 'Y']] = self.data[['X', 'Y']].apply(
                lambda row: pd.Series([
                    np.dot([row.iloc[0] - origin[0], row.iloc[1] - origin[1]], Uu),
                    np.dot([row.iloc[0] - origin[0], row.iloc[1] - origin[1]], Vu)
                ]), axis=1)

            self.data['X'] = (self.data['X'] / x_res).round().astype(int)
            self.data['Y'] = (self.data['Y'] / y_res).round().astype(int)
            self.data['Z'] = (self.data['Z'] / z_res).round().astype(int)

            bounds = {
                'x_min': int(self.data['X'].min()),
                'x_max': int(self.data['X'].max()),
                'y_min': int(self.data['Y'].min()),
                'y_max': int(self.data['Y'].max()),
                'z_min': int(self.data['Z'].min()),
                'z_max': int(self.data['Z'].max()),
            }

            self.logger.info(
                f"Coordinates aligned; "
                f"X[{bounds['x_min']},{bounds['x_max']}] "
                f"Y[{bounds['y_min']},{bounds['y_max']}] "
                f"Z[{bounds['z_min']},{bounds['z_max']}]"
            )
            return bounds
        
        except Exception as e:
            self.logger.error(f"Error aligning coordinates: {str(e)}")
            return None, None, None, None, None, None
    
    def create_param_dict(self):
        """
        Create parameter dictionaries for all available parameters.
        
        Returns:
        --------
        dict
            Dictionary of parameter values
        """
        try:
            # Get all parameter columns
            param_cols = self.data.columns.difference(['Borehole ID', 'Borehole type', 'X', 'Y', 'Z'])
            
            # Create dictionary for each parameter
            self.params_dict = {
                param: {
                    (r.X, r.Y, r.Z): r[param] 
                    for _, r in self.data[['X', 'Y', 'Z', param]].dropna().iterrows()
                }
                for param in param_cols
            }
            
            self.logger.info(f"Parameter dictionaries created for {len(param_cols)} parameters")
            return self.params_dict
            
        except Exception as e:
            self.logger.error(f"Error creating parameter dictionaries: {str(e)}")
            return {}
    
    def generate_grid(self, bounds, r):
        """
        Generate 3D grid for the study area.

        Parameters
        ----------
        bounds : dict
            {'x_min','x_max','y_min','y_max','z_min','z_max'}
        r : float
            Neighbor radius for edges

        Returns
        -------
        tuple
            (coordinates[int64 Nx3], labels[int64 N], edge_index[int64 2xE])
        """
        try:
            if not self.params_dict:
                self.create_param_dict()

            x_min = int(bounds['x_min']); x_max = int(bounds['x_max'])
            y_min = int(bounds['y_min']); y_max = int(bounds['y_max'])
            z_min = int(bounds['z_min']); z_max = int(bounds['z_max'])
            
            # Create 3D grid
            X, Y, Z = np.meshgrid(
                np.arange(x_min, x_max + 1),
                np.arange(y_min, y_max + 1),
                np.arange(z_min, z_max + 1),
                indexing='ij'
            )
            
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.int64)
            labels = -1 * np.ones(len(coords), dtype=np.int64)
            
            # Assign known values from boreholes
            soil_key = 'Soil type'
            if soil_key in self.params_dict:
                for (x, y, z), v in self.params_dict[soil_key].items():
                    idx = np.where((coords[:, 0] == x) & (coords[:, 1] == y) & (coords[:, 2] == z))
                    if idx[0].size > 0:
                        labels[idx] = int(v)
            else:
                self.logger.warning(f"'{soil_key}' not found in params_dict; all labels remain -1")
            
            # Create edge connections (graph structure)
            tree = cKDTree(coords.astype(np.float64))
            edges = tree.query_pairs(r=float(r), output_type='ndarray')
            edge_index = edges.T
            
            self.logger.info(
                f"3D grid generated with {len(coords)} points and {len(edges)} connections "
                f"(x:[{x_min},{x_max}] y:[{y_min},{y_max}] z:[{z_min},{z_max}] r={r})"
            )
            return coords, labels, edge_index
            
        except Exception as e:
            self.logger.error(f"Error generating grid: {str(e)}")
            return None, None, None