"""
TerranGraph - Command Line Interface
=================================

Command line interface for TerranGraph framework.
"""

import argparse
import sys
import logging
from pathlib import Path

from .core.data_handler import DataHandler
from .core.geo_model import GeoModel  
from .core.visualizer import GeoVisualizer
from . import __version__

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='TerranGraph - Graph-Based Geotechnical Site Representation',
        prog='terrangraph'
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    gui_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train geological model')
    train_parser.add_argument('data_file', help='Input data file (Excel/CSV)')
    train_parser.add_argument('--output', '-o', help='Output model file', default='model.pth')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    train_parser.add_argument('--hidden-size', type=int, default=32, help='Hidden layer size')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--x-res', type=float, default=10, help='X resolution')
    train_parser.add_argument('--y-res', type=float, default=2, help='Y resolution')
    train_parser.add_argument('--z-res', type=float, default=1, help='Z resolution')
    train_parser.add_argument('--z-min', type=int, default=-64, help='Minimum Z value')
    train_parser.add_argument('--z-max', type=int, default=8, help='Maximum Z value')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('model_file', help='Trained model file')
    predict_parser.add_argument('data_file', help='Input data file')
    predict_parser.add_argument('--output', '-o', help='Output file', default='predictions.csv')
    predict_parser.add_argument('--format', choices=['csv', 'xlsx'], default='csv', help='Output format')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize model')
    viz_parser.add_argument('model_file', help='Model file')
    viz_parser.add_argument('data_file', help='Data file')
    viz_parser.add_argument('--export', help='Export visualization to file')
    viz_parser.add_argument('--format', choices=['ply', 'obj', 'stl'], default='ply', help='Export format')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data file')
    validate_parser.add_argument('data_file', help='Data file to validate')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser

def train_model(args):
    """Train model command."""
    
    print(f"Training model with data from {args.data_file}")
    
    # Load data
    handler = DataHandler()
    if not handler.load_data(args.data_file):
        print(f"Failed to load data from {args.data_file}")
        return 1
    
    print(f"Loaded {len(handler.data)} data points with {len(handler.soil_map)} soil types")
    
    # Process data
    print("Processing data...")
    handler.compute_mbr()
    handler.align_coordinates(args.x_res, args.y_res, args.z_res)
    coords, labels, edges = handler.generate_grid(args.z_min, args.z_max)
    
    if coords is None:
        print("Failed to generate grid")
        return 1
    
    print(f"Generated grid with {len(coords)} points and {len(edges.T)} edges")
    
    # Train model
    print("Building and training model...")
    model = GeoModel()
    model.prepare_data(coords, labels, edges)
    num_classes = len(handler.soil_map)
    model.build_model(num_classes, args.hidden_size)
    
    print(f"Training for {args.epochs} epochs...")
    history = model.train(epochs=args.epochs, lr=args.lr)
    
    # Evaluate model
    metrics = model.evaluate()
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    
    # Save model
    if model.save_model(args.output):
        print(f"Model saved to {args.output}")
    else:
        print("Failed to save model")
        return 1
    
    return 0

def predict_with_model(args):
    """Generate predictions command."""
    
    print(f"Generating predictions with model {args.model_file}")
    
    # Load data
    handler = DataHandler()
    if not handler.load_data(args.data_file):
        print(f"Failed to load data from {args.data_file}")
        return 1
    
    # Process data (simplified - reuse training parameters)
    handler.compute_mbr()
    handler.align_coordinates()
    coords, labels, edges = handler.generate_grid()
    
    # Load model
    model = GeoModel()
    num_classes = len(handler.soil_map)
    if not model.load_model(args.model_file, num_classes):
        print(f"Failed to load model from {args.model_file}")
        return 1
    
    # Prepare data and predict
    model.prepare_data(coords, labels, edges)
    predictions, probabilities = model.predict()
    
    if predictions is None:
        print("Failed to generate predictions")
        return 1
    
    # Save predictions
    import pandas as pd
    pred_df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1], 
        'Z': coords[:, 2],
        'Predicted_Class': predictions,
        'Max_Probability': probabilities.max(axis=1)
    })
    
    # Add class probabilities
    for i in range(num_classes):
        pred_df[f'Prob_Class_{i}'] = probabilities[:, i]
    
    if args.format == 'xlsx':
        pred_df.to_excel(args.output, index=False)
    else:
        pred_df.to_csv(args.output, index=False)
    
    print(f"Predictions saved to {args.output}")
    return 0

def visualize_model(args):
    """Visualize model command."""
    
    print(f"Visualizing model {args.model_file}")
    
    # Load data
    handler = DataHandler()
    if not handler.load_data(args.data_file):
        print(f"Failed to load data from {args.data_file}")
        return 1
    
    # Process data
    handler.compute_mbr()
    handler.align_coordinates()
    coords, labels, edges = handler.generate_grid()
    
    # Load model and predict
    model = GeoModel()
    num_classes = len(handler.soil_map)
    if not model.load_model(args.model_file, num_classes):
        print(f"Failed to load model from {args.model_file}")
        return 1
    
    model.prepare_data(coords, labels, edges)
    predictions, _ = model.predict()
    
    # Create 3D model
    visualizer = GeoVisualizer()
    mesh = visualizer.create_model(coords, predictions)
    
    if args.export:
        if visualizer.export_model(args.export, mesh, args.format):
            print(f"3D model exported to {args.export}")
        else:
            print("Failed to export model")
            return 1
    else:
        # Show interactive visualization
        if not visualizer.show_model(mesh):
            print("Failed to display 3D model")
            return 1
    
    return 0

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'predict':
        return predict_with_model(args)
    elif args.command == 'visualize':
        return visualize_model(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())