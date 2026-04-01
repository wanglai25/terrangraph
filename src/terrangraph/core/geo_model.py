"""
TerranGraph - Graph-Based Site Representation Module
==================================================

Graph neural network module for geotechnical site representation and prediction.

This module implements:
- Graph construction-based learning from spatial grids
- GNN-based classification of subsurface materials
- Training, inference, and evaluation pipelines
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
    
class GNN(nn.Module):
    """Graph Neural Network for geotechnical classification."""
    def __init__(self, num_classes, in_channels, hidden=32, gcn_layers=3, mlp_layers=4, dropout=0.1):
        super().__init__()
        # GCN stacks
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = in_channels if i == 0 else hidden
            self.convs.append(GCNConv(in_dim, hidden))
            self.bns.append(BatchNorm(hidden))
        # MLP head
        layers = []
        for _ in range(mlp_layers):
            layers += [
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(hidden, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        return self.head(x)

class GeoModel:
    """
    Graph-based model for geotechnical site representation.

    This class provides:
    - Data preparation
    - Model construction
    - Training and evaluation
    - Prediction and persistence
    """
    
    def __init__(self, device=None):
        self.model = None
        self.data = None
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_history = {'loss': [], 'accuracy': []}
        self.logger = logging.getLogger(__name__ + ".GeoModel")
        self.logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, coords, labels, edge_index):
        """
        Prepare data for GNN model.
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinates of grid points
        labels : np.ndarray
            Labels (soil types) for grid points
        edge_index : np.ndarray
            Edges connecting grid points
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            features = torch.tensor(np.column_stack([coords, labels]), dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
            
            self.data = Data(x=features, edge_index=edge_index).to(self.device)
            self.mask = (features[:, 3] != -1).to(self.device)
            
            self.logger.info(f"Data prepared: {len(features)} nodes, {edge_index.shape[1]} edges")
            self.logger.info(f"Known points: {self.mask.sum().item()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return False
        
    def build_model(self, num_classes, hidden_size=32, gcn_layers=3, mlp_layers=3, dropout=0.1):
        """
        Build GNN with structure customization:
        - hidden_size: hidden units
        - gcn_layers : number of GCN layers
        - mlp_layers : number of MLP blocks
        - dropout    : dropout ratio in MLP
        """
        try:
            in_channels = int(self.data.x.size(1)) if self.data is not None else 4
            self.model = GNN(
                num_classes=num_classes,
                in_channels=in_channels,
                hidden=hidden_size,
                gcn_layers=gcn_layers,
                mlp_layers=mlp_layers,
                dropout=dropout
            ).to(self.device)

            self.logger.info(
                f"Model built: in={in_channels}, hidden={hidden_size}, "
                f"gcn_layers={gcn_layers}, mlp_layers={mlp_layers}, dropout={dropout}"
            )
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Model has {num_params} trainable parameters")
            return True
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            return False
    
    def train(self, epochs=1000, lr=0.01, validation_interval=None):
        """ 
        Train the GNN model. 

        Parameters: 
        ----------- 
        epochs : int 
            Number of training epochs 
        lr : float 
            Learning rate 
        validation_interval : 
        int 
            Interval for validation and logging 

        Returns: 
        -------- 
        dict 
            Training history 
        """
        try:
            if self.model is None or self.data is None:
                self.logger.error("Model or data not initialized")
                return self.train_history

            if validation_interval is None:
                validation_interval = max(1, epochs // 10)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
            loss_fn = torch.nn.CrossEntropyLoss()

            self.train_history = {'loss': [], 'accuracy': []}

            for epoch in range(1, epochs + 1):
                self.model.train()
                optimizer.zero_grad()

                out = self.model(self.data.x, self.data.edge_index)
                loss = loss_fn(out[self.mask], self.data.x[self.mask, 3].long())
                loss.backward()
                optimizer.step()
                scheduler.step(loss.item())

                self.train_history['loss'].append(loss.item())

                if (epoch % validation_interval == 0) or (epoch == 1) or (epoch == epochs):
                    self.model.eval()
                    with torch.no_grad():
                        preds = self.model(self.data.x, self.data.edge_index).argmax(dim=1).cpu().numpy()
                    true = self.data.x[:, 3].cpu().numpy()
                    valid = true != -1
                    acc = accuracy_score(true[valid], preds[valid]) if valid.sum() > 0 else float('nan')
                    self.train_history['accuracy'].append(acc)

                    # 当前学习率
                    curr_lr = optimizer.param_groups[0].get("lr", lr)
                    self.logger.info(f"Epoch {epoch:04d}/{epochs}: Loss={loss.item():.4f} Acc={acc:.4f} LR={curr_lr:.6f}")

            self.logger.info("Training completed")
            return self.train_history

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return self.train_history

    def predict(self):
        """
        Generate predictions for the entire grid.
        
        Returns:
        --------
        tuple
            (predicted classes, probabilities)
        """
        try:
            if self.model is None or self.data is None:
                self.logger.error("Model or data not initialized")
                return None, None
            
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data.x, self.data.edge_index)
                preds = out.argmax(dim=1).cpu().numpy()
                probs = F.softmax(out, dim=1).cpu().numpy()
            
            self.logger.info("Predictions generated")
            return preds, probs
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return None, None
    
    def evaluate(self):
        """
        Evaluate model performance.
    
        Returns:
        --------
        dict
            Evaluation metrics
        """
        try:
            if self.model is None or self.data is None:
                self.logger.error("Model or data not initialized")
                return {}
        
            # Generate predictions
            preds, _ = self.predict()
        
            # Get ground truth
            true = self.data.x[:, 3].cpu().numpy()
            valid = true != -1
        
            # Calculate metrics
            acc = accuracy_score(true[valid], preds[valid])
            conf_mat = confusion_matrix(true[valid], preds[valid])
            report = classification_report(true[valid], preds[valid], output_dict=True, zero_division=0)
        
            metrics = {
                'accuracy': acc,
                'confusion_matrix': conf_mat,
                'classification_report': report
            }
        
            self.logger.info(f"Model evaluation - Accuracy: {acc:.4f}")
        
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def save_model(self, path):
        """
        Save model to file.
        
        Parameters:
        -----------
        path : str
            Path to save model
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            if self.model is None:
                self.logger.error("Model not initialized")
                return False
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'train_history': self.train_history
            }, path)
            
            self.logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path, num_classes, hidden_size=32):
        """
        Load model from file.
        
        Parameters:
        -----------
        path : str
            Path to model file
        num_classes : int
            Number of soil classes
        hidden_size : int
            Size of hidden layers
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            # Build model architecture
            self.build_model(num_classes, hidden_size)
            
            # Load state dict
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_history = checkpoint['train_history']
            
            self.logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False