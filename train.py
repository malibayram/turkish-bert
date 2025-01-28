import torch
from torch.utils.data import DataLoader, random_split
from bert.dataset import BERTDataset, collate_batch
from turkish_tokenizer.turkish_tokenizer import TurkishTokenizer
from bert import BERT, BERTLM, BERTTrainer
import argparse
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train Turkish BERT model')
    
    # Data parameters
    parser.add_argument('--corpus_path', type=str, default='combined_reviews.csv',
                        help='Path to the training corpus')
    parser.add_argument('--train_test_split', type=float, default=0.9,
                        help='Proportion of data to use for training')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=32768,
                        help='Size of vocabulary')
    parser.add_argument('--d_model', type=int, default=768,
                        help='Dimension of model')
    parser.add_argument('--n_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help='Number of warmup steps')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to use for training')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save model every N epochs')
    
    return parser.parse_args()

def setup_device(args):
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        # Adjust batch size and other parameters for MPS
        if args.batch_size > 32:
            print("Warning: Reducing batch size to 32 for MPS device")
            args.batch_size = 32
        if args.seq_len > 512:
            print("Warning: Reducing sequence length to 512 for MPS device")
            args.seq_len = 512
    else:
        device = 'cpu'
    
    # Set default num_workers based on device
    if args.num_workers is None:
        if device == 'mps':
            args.num_workers = 0  # Use 0 workers with MPS
        else:
            args.num_workers = min(4, os.cpu_count() or 1)
            
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of workers: {args.num_workers}")
    return device

def main():
    args = parse_args()
    device = setup_device(args)
    
    # Create directory for checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Initialize tokenizer
        tokenizer = TurkishTokenizer()
        
        # Create dataset with error handling
        try:
            dataset = BERTDataset(
                corpus_path=args.corpus_path,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                device=device
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        # Create train/test split
        train_size = int(args.train_test_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create dataloaders with error handling
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=device != 'cpu',  # Use pin_memory except for CPU
            persistent_workers=args.num_workers > 0,  # Keep workers alive between batches
            drop_last=True  # Drop incomplete batches
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=device != 'cpu',
            persistent_workers=args.num_workers > 0,
            drop_last=True
        )
        
        # Print dataset information
        print(f"Total samples: {len(dataset)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
        # Create models
        bert = BERT(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            heads=args.heads,
            dropout=args.dropout,
            seq_len=args.seq_len
        )
        
        bert_lm = BERTLM(
            bert=bert,
            vocab_size=args.vocab_size
        )
        
        # Create trainer
        trainer = BERTTrainer(
            model=bert_lm,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            device=device
        )
        
        # Training loop with error handling
        best_val_loss = float('inf')
        for epoch in range(args.num_epochs):
            try:
                # Training phase
                train_metrics = trainer.train(epoch)
                
                # Validation phase
                val_metrics = trainer.test(epoch)
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Update training monitor
                trainer.monitor.update(metrics, epoch)
                
                # Check for best model
                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                
                # Save checkpoint
                if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
                    trainer.checkpointing.save(
                        model=bert_lm,
                        optimizer=trainer.optim,
                        scheduler=trainer.optim_schedule,
                        epoch=epoch,
                        val_loss=val_metrics['val_loss'],
                        metrics=metrics,
                        args=args,
                        is_best=is_best
                    )
                    
                # Early stopping check
                trainer.early_stopping(val_metrics['val_loss'])
                if trainer.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print(f"ERROR: {str(e)}")
                    raise e
                    
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save final checkpoint
        save_path = f"{args.save_dir}/bert_interrupted.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': bert_lm.state_dict(),
            'optimizer_state_dict': trainer.optim.state_dict(),
            'optimizer_params': {
                'warmup_steps': trainer.optim_schedule.n_warmup_steps,
                'init_lr': trainer.optim_schedule.init_lr,
            },
            'args': args.__dict__,
        }
        torch.save(checkpoint, save_path)
        print(f"Saved interrupt checkpoint to {save_path}")

if __name__ == "__main__":
    main()