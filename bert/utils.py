import torch
import json
from pathlib import Path

class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.metrics_dir = self.save_dir / 'metrics'
        self.logs_dir = self.save_dir / 'logs'
        
        for dir in [self.metrics_dir, self.logs_dir]:
            dir.mkdir(parents=True, exist_ok=True)
            
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'train_mlm_acc': [],
            'train_nsp_acc': [],
            'val_mlm_acc': [],
            'val_nsp_acc': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def update(self, metrics, epoch):
        print("\n" + "="*50)
        print(f"Training Report - Epoch {epoch}")
        print("="*50)
        
        # Update and save current epoch metrics
        for k, v in metrics.items():
            try:
                value = float(v)
                self.metrics[k].append(value)
                print(f"{k}: {value:.6f}")
            except (TypeError, ValueError) as e:
                print(f"Error with metric {k}: {e}")
        
        # Save detailed epoch report
        epoch_report = {
            'epoch': epoch,
            'metrics': metrics,
            'cumulative_stats': {
                k: {
                    'current': v[-1],
                    'best': min(v) if 'loss' in k else max(v),
                    'average': sum(v) / len(v),
                    'history': v
                } for k, v in self.metrics.items()
            }
        }
        
        # Save epoch report
        metrics_file = self.metrics_dir / f'epoch_{epoch}_report.json'
        with open(metrics_file, 'w') as f:
            json.dump(epoch_report, f, indent=4)
        
        # Save cumulative metrics with statistics
        stats_report = {
            'current_epoch': epoch,
            'metrics_summary': {
                k: {
                    'current': v[-1],
                    'min': min(v),
                    'max': max(v),
                    'average': sum(v) / len(v),
                    'improvement': v[-1] - v[0],  # Positive means improvement
                    'best_epoch': v.index(min(v)) if 'loss' in k else v.index(max(v))
                } for k, v in self.metrics.items()
            },
            'training_progress': {
                'total_epochs': epoch + 1,
                'best_val_loss': min(self.metrics['val_loss']),
                'best_val_loss_epoch': self.metrics['val_loss'].index(min(self.metrics['val_loss'])),
                'best_mlm_acc': max(self.metrics['train_mlm_acc']),
                'best_nsp_acc': max(self.metrics['train_nsp_acc'])
            }
        }
        
        # Save summary report
        with open(self.metrics_dir / 'training_summary.json', 'w') as f:
            json.dump(stats_report, f, indent=4)
        
        # Save training log
        log_message = (
            f"\nEpoch {epoch} Summary:\n"
            f"Training Loss: {metrics['train_loss']:.6f}\n"
            f"Validation Loss: {metrics['val_loss']:.6f}\n"
            f"MLM Accuracy: {metrics['train_mlm_acc']:.2%}\n"
            f"NSP Accuracy: {metrics['train_nsp_acc']:.2%}\n"
            f"Learning Rate: {metrics['learning_rates']:.2e}\n"
            f"Best Val Loss So Far: {min(self.metrics['val_loss']):.6f}\n"
            f"Best MLM Accuracy: {max(self.metrics['train_mlm_acc']):.2%}\n"
            f"Best NSP Accuracy: {max(self.metrics['train_nsp_acc']):.2%}\n"
            f"{'-'*50}\n"
        )
        
        with open(self.logs_dir / 'training_log.txt', 'a') as f:
            f.write(log_message)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Checkpointing:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
    def save(self, model, optimizer, scheduler, epoch, val_loss, metrics, args, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_params': {
                'warmup_steps': scheduler.n_warmup_steps,
                'init_lr': scheduler.init_lr,
            },
            'val_loss': val_loss,
            'metrics': metrics,
            'args': args.__dict__
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            
        # Save periodic checkpoint
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        
    def load(self, path):
        return torch.load(path) 