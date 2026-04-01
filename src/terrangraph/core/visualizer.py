"""
TerranGraph - Visualization Module
===============================

Visualization module for TerranGraph.

This module provides:
- 3D subsurface model reconstruction
- Interactive visualization using PyVista
- Cross-section extraction
- 2D property mapping for spatial analysis
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import builtins
import pyvista as pv

class GeoVisualizer:
    """
    Visualization module for geotechnical site representation.

    Supports 3D model reconstruction, slicing, and property mapping.
    """
    
    def __init__(self):
        self.mesh = None         # pv.DataSet (PolyData/UnstructuredGrid)
        self.grid = None         # pv.UniformGrid with cell scalars "class"
        self.logger = logging.getLogger(__name__ + ".GeoVisualizer")

    @staticmethod
    def _build_image_data_from_coords(coords, classes, x_res=10, y_res=2, z_res=1):
        """Build ImageData from coordinates and classes."""
        coords = np.asarray(coords, dtype=float)
        classes = np.asarray(classes)

        xs = np.unique(coords[:, 0])
        ys = np.unique(coords[:, 1])
        zs = np.unique(coords[:, 2])
        nx, ny, nz = len(xs), len(ys), len(zs)

        if classes.size != nx * ny * nz:
            raise ValueError(f"Classes length {classes.size} != nx*ny*nz = {nx*ny*nz}")

        try:
            grid = pv.ImageData()
        except AttributeError:
            import vtk
            grid = pv.wrap(vtk.vtkImageData())

        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (float(x_res), float(y_res), float(z_res))

        origin = (
            float(xs.min() * x_res - 0.5 * x_res),
            float(ys.min() * y_res - 0.5 * y_res),
            float(zs.min() * z_res - 0.5 * z_res),
        )
        grid.origin = origin

        vals = classes.reshape((nx, ny, nz), order="C")
        grid.cell_data["class"] = vals.ravel(order="F").astype(float)

        return grid
   
    def _cmap_from_class_colors(self, class_colors):
        """Create colormap from class colors."""
        cc = np.asarray(class_colors, dtype=float)
        if cc.max() > 1.0:   # Handle 0-255 range
            cc = cc / 255.0
        return ListedColormap(cc)

    def create_model(self, coords, classes, class_colors=None, x_res=10, y_res=2, z_res=1):
        """
        Create a 3D geological model as a surface mesh (PolyData).
        """
        try:
            # Build the uniform grid with cell scalar "class"
            self.grid = self._build_image_data_from_coords(coords, classes, x_res, y_res, z_res)

            # Determine valid classes (>= 0)
            defined = classes >= 0
            if not np.any(defined):
                self.logger.warning("No defined classes (>=0) found; returning empty mesh.")
                self.mesh = pv.PolyData()
                return self.mesh

            unique_classes = np.unique(classes[defined]).astype(int)

            # Merge class-specific surfaces
            merged = None
            for k in unique_classes:
                # Extract cells for class k by thresholding
                sub = self.grid.threshold([k - 0.5, k + 0.5], scalars="class")
                if sub.n_cells == 0:
                    continue

                # Extract surface geometry
                surf = sub.extract_geometry()
                surf.cell_data["class"] = np.full(surf.n_cells, k, dtype=float)

                if merged is None:
                    merged = surf.copy()
                else:
                    merged = merged.merge(surf)

            if merged is None:
                self.mesh = pv.PolyData()
            else:
                self.mesh = merged

            self.logger.info(f"3D model created: {self.mesh.n_cells:,} surface cells, classes: {unique_classes.tolist()}")
            return self.mesh

        except Exception as e:
            self.logger.error(f"Error creating 3D model: {str(e)}")
            return None

    def show_model(self, mesh=None, class_colors=None, title="GeoTwinNet - 3D Site Model"):
        """Show the 3D model with PyVista's interactive Plotter."""
        try:
            if mesh is None:
                mesh = self.mesh
            if mesh is None or mesh.n_cells == 0:
                self.logger.error("No mesh to visualize")
                return False

            p = pv.Plotter(window_size=(1000, 800))
            p.add_axes()
            p.add_bounding_box()

            if "class" in mesh.cell_data:
                if class_colors is not None:
                    cmap = self._cmap_from_class_colors(class_colors)
                    n = len(class_colors)
                    p.add_mesh(
                        mesh, scalars="class", categories=True, cmap=cmap,
                        clim=[-0.5, n - 0.5], show_scalar_bar=False
                    )
                else:
                    p.add_mesh(mesh, scalars="class", categories=True, cmap="tab20", show_scalar_bar=False)
            else:
                p.add_mesh(mesh, color="lightgray")

            p.show(interactive=True, auto_close=False)

            self.logger.info(f"3D model visualized")
            return True

        except Exception as e:
            self.logger.error(f"Error visualizing 3D model: {str(e)}")
            return False

    def export_model(self, path, mesh=None, format='ply'):
        """Export 3D model to file."""
        try:
            if mesh is None:
                mesh = self.mesh

            if mesh is None or mesh.n_cells == 0:
                self.logger.error("No mesh to export")
                return False

            fmt = str(format).lower()
            if not path.lower().endswith(f".{fmt}"):
                path = f"{path}.{fmt}"

            # Handle GLB format limitation
            if fmt == "glb":
                self.logger.warning("GLB export is not supported via PyVista. Exporting OBJ instead.")
                alt_path = path[:-4] + ".obj"
                mesh.save(alt_path)
                self.logger.info(f"3D model exported to {alt_path} (OBJ)")
                return True

            mesh.save(path)
            self.logger.info(f"3D model exported to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting 3D model: {str(e)}")
            return False

    def create_cross_section(self, mesh, plane_origin, plane_normal):
        """Create cross-section of the 3D model by slicing with a plane."""
        try:
            if mesh is None:
                mesh = self.mesh

            if mesh is None or mesh.n_cells == 0:
                self.logger.error("No mesh for cross-section")
                return None, None

            # Normalize normal
            n = np.asarray(plane_normal, dtype=float)
            n /= np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0

            # Perform slice
            dataset = self.grid if self.grid is not None else mesh
            sl = dataset.slice(origin=np.asarray(plane_origin, dtype=float), normal=n)

            # Extract surface if needed
            if not isinstance(sl, pv.PolyData):
                sl = sl.extract_geometry()

            self.logger.info("Cross-section created")
            return sl, dataset

        except Exception as e:
            self.logger.error(f"Error creating cross-section: {str(e)}")
            return None, None

    def plot_property_map(self, coords, values, class_colors=None, property_name="", slice_dim='z', slice_value=0):
        """Create 2D property map for a specific slice."""
        try:
            # Create dataframe
            df = pd.DataFrame(np.column_stack([coords, values]), columns=['X', 'Y', 'Z', 'Value'])

            # Select slice
            if slice_dim.lower() == 'x':
                df_slice = df[df['X'] == slice_value]
                x_col, y_col = 'Y', 'Z'
                title = f"{property_name} (X = {slice_value})"
            elif slice_dim.lower() == 'y':
                df_slice = df[df['Y'] == slice_value]
                x_col, y_col = 'X', 'Z'
                title = f"{property_name} (Y = {slice_value})"
            else:  # z
                df_slice = df[df['Z'] == slice_value]
                x_col, y_col = 'X', 'Y'
                title = f"{property_name} (Z = {slice_value})"

            if len(df_slice) == 0:
                self.logger.warning(f"No data points found at {slice_dim} = {slice_value}")
                return None

            fig, ax = plt.subplots(figsize=(12, 10))

            if class_colors is not None:
                # Map class indices to RGB colors
                colors = []
                for v in df_slice['Value'].values:
                    if v >= 0 and int(v) < len(class_colors):
                        colors.append(class_colors[int(v)])
                    else:
                        colors.append([0.8, 0.8, 0.8])
                ax.scatter(df_slice[x_col], df_slice[y_col], c=colors, s=100, alpha=0.8)
            else:
                sc = ax.scatter(df_slice[x_col], df_slice[y_col], c=df_slice['Value'], cmap='viridis', s=100, alpha=0.8)
                plt.colorbar(sc, ax=ax, label=property_name)

            ax.set_xlabel(f'{x_col} Coordinate')
            ax.set_ylabel(f'{y_col} Coordinate')
            ax.set_title(title)
            ax.grid(True)
            ax.set_aspect('equal')

            self.logger.info(f"Property map created for {property_name}")
            return fig

        except Exception as e:
            self.logger.error(f"Error creating property map: {str(e)}")
            return None