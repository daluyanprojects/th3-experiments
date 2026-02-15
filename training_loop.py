import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
import csv
import  matplotlib.pyplot as plt

from training import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_training_history
)

class Trainer:    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        device: torch.device,
        config: Dict,
        scaler: Optional[object] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.scaler = scaler
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }
        
        # Setup directories
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, (spatial, rainfall, labels) in enumerate(pbar):
            # Move to device
            spatial = spatial.to(self.device)
            rainfall = rainfall.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(spatial, rainfall)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip_norm']
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits, _ = self.model(spatial, rainfall)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip_norm']
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for spatial, rainfall, labels in pbar:
                # Move to device
                spatial = spatial.to(self.device)
                rainfall = rainfall.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(spatial, rainfall)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item() * labels.size(0)
                _, predicted = logits.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, self.config['num_classes'])
        metrics['loss'] = total_loss / len(all_labels)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: Optional[int] = None):
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "="*70)
        print("PHASE 11: TRAINING LOOP")
        print("="*70)
        print(f"Training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['macro_f1'])
            self.history['lr'].append(current_lr)
            
            # Console logging
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['macro_f1']
                self.epochs_without_improvement = 0
                
                print(f"  ✓ New best model! Val Acc: {self.best_val_acc:.4f}")
                self.save_checkpoint('best_model.pth', val_metrics)
            else:
                self.epochs_without_improvement += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', val_metrics)
            
            if self.epochs_without_improvement >= self.config.get('early_stop_patience', 30):
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
                break

        total_time = time.time() - start_time
        
        # Save final model
        self.save_checkpoint('final_model.pth', val_metrics)
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("="*70 + "\n")
    
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'config': self.config,
            'metrics': metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if 'best' in filename:
            print(f"    Saved to: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)   
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best val acc: {self.best_val_acc:.4f}")
    
    def save_history(self):        
        history_path = self.log_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"  Training history saved: {history_path}")
    
    def plot_training_curves(self):        
        fig = plot_training_history(self.history,save_path=self.log_dir / 'training_curves.png') 
        plt.close(fig)
    

    def test(self, test_loader: DataLoader, split_name: str = 'Test') -> Dict:
        print("\n" + "="*70)
        print(f"PHASE 12: {split_name.upper()} EVALUATION")
        print("="*70)
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'{split_name} Evaluation')
            
            for spatial, rainfall, labels in pbar:
                spatial = spatial.to(self.device)
                rainfall = rainfall.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(spatial, rainfall)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_probs = np.vstack(all_probs)
        
        # Compute comprehensive metrics
        metrics = compute_metrics(y_true, y_pred, self.config['num_classes'])
        
        # Print metrics
        print_metrics(metrics, split=split_name)
        
        # Plot confusion matrix
        fig = plot_confusion_matrix(
            metrics['confusion_matrix'],
            title=f'{split_name} Confusion Matrix',
            save_path=self.log_dir / f'{split_name.lower()}_confusion_matrix.png'
        )
        plt.close(fig)
        
        # Save detailed results
        results = {
            'metrics': {k: float(v) if isinstance(v, (np.number, float)) else v 
                       for k, v in metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'predictions': y_pred.tolist(),
            'labels': y_true.tolist(),
            'probabilities': y_probs.tolist()
        }
        
        results_path = self.log_dir / f'{split_name.lower()}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  Results saved to: {results_path}")
        print("="*70 + "\n")
        
        return metrics


def create_dataloaders(X_train_spatial: np.ndarray, X_train_rainfall: np.ndarray,
    y_train: np.ndarray, X_val_spatial: np.ndarray, X_val_rainfall: np.ndarray,
    y_val: np.ndarray, batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_spatial),
        torch.FloatTensor(X_train_rainfall),
        torch.LongTensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_spatial),
        torch.FloatTensor(X_val_rainfall),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader