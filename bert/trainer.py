import torch
from torch.optim import Adam
from tqdm import tqdm
from bert.scheduler import ScheduledOptim
from .utils import TrainingMonitor, EarlyStopping, Checkpointing
import torch.nn.functional as F

class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device=None,
        save_dir='checkpoints',
        patience=3,
        gradient_clip_val=1.0
        ):
        """
        Initialize BERTTrainer.
        
        Args:
            model: BERT model to train
            train_dataloader: training data loader
            test_dataloader: test/validation data loader
            lr: learning rate
            weight_decay: weight decay for regularization
            betas: Adam optimizer betas
            warmup_steps: number of warmup steps for learning rate
            log_freq: logging frequency
            device: device to train on ('cuda', 'mps', or 'cpu')
            save_dir: directory to save checkpoints
            patience: patience for early stopping
            gradient_clip_val: gradient clipping value
        """
        # Determine the device to use
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")
        
        # Move model to the appropriate device
        self.model = model.to(self.device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=8)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        self.monitor = TrainingMonitor(save_dir)
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpointing = Checkpointing(save_dir)
        self.gradient_clip_val = gradient_clip_val
    
    def train(self, epoch):
        self.model.train()
        total_loss = 0
        total_mlm_correct = 0
        total_nsp_correct = 0
        total_tokens = 0
        total_sequences = 0
        
        data_iter = tqdm(
            enumerate(self.train_data),
            desc=f"EP_{epoch}",
            total=len(self.train_data),
            bar_format="{l_bar}{r_bar}"
        )
        
        for i, data in data_iter:
            # Move data to device
            data = {key: value.to(self.device) for key, value in data.items()}
            
            # Forward pass
            next_sent_output, mask_lm_output = self.model.forward(
                data["bert_input"], 
                data["segment_label"]
            )
            
            # Calculate losses
            next_loss = self.criterion(next_sent_output, data["is_next"])
            mask_loss = self.criterion(
                mask_lm_output.transpose(1, 2),
                data["bert_label"]
            )
            loss = next_loss + mask_loss
            
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optim.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracies
            mlm_correct = (mask_lm_output.argmax(dim=-1) == data["bert_label"]).sum().item()
            nsp_correct = (next_sent_output.argmax(dim=-1) == data["is_next"]).sum().item()
            
            total_mlm_correct += mlm_correct
            total_nsp_correct += nsp_correct
            total_tokens += (data["bert_label"] != -100).sum().item()
            total_sequences += data["is_next"].size(0)
            
            # Update progress bar
            current_lr = self.optim_schedule.get_lr()
            data_iter.set_description(
                f"EP_{epoch} - loss: {loss.item():.4f}, lr: {current_lr:.2e}"
            )
            
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_data)
        mlm_accuracy = total_mlm_correct / total_tokens if total_tokens > 0 else 0.0
        nsp_accuracy = total_nsp_correct / total_sequences if total_sequences > 0 else 0.0
        current_lr = self.optim_schedule.get_lr()
        
        metrics = {
            'train_loss': float(epoch_loss),
            'train_mlm_acc': float(mlm_accuracy),
            'train_nsp_acc': float(nsp_accuracy),
            'learning_rates': float(current_lr)
        }
        
        print(f"\nTraining metrics for epoch {epoch}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        
        return metrics

    def test(self, epoch):
        self.model.eval()
        total_loss = 0
        total_mlm_correct = 0
        total_nsp_correct = 0
        total_tokens = 0
        total_sequences = 0
        
        with torch.no_grad():
            for data in self.test_data:
                data = {key: value.to(self.device) for key, value in data.items()}
                next_sent_output, mask_lm_output = self.model.forward(
                    data["bert_input"],
                    data["segment_label"]
                )
                
                next_loss = self.criterion(next_sent_output, data["is_next"])
                mask_loss = self.criterion(
                    mask_lm_output.transpose(1, 2),
                    data["bert_label"]
                )
                loss = next_loss + mask_loss
                
                total_loss += loss.item()
                
                mlm_correct = (mask_lm_output.argmax(dim=-1) == data["bert_label"]).sum().item()
                nsp_correct = (next_sent_output.argmax(dim=-1) == data["is_next"]).sum().item()
                
                total_mlm_correct += mlm_correct
                total_nsp_correct += nsp_correct
                total_tokens += (data["bert_label"] != -100).sum().item()
                total_sequences += data["is_next"].size(0)
                
        val_loss = total_loss / len(self.test_data)
        mlm_accuracy = total_mlm_correct / total_tokens if total_tokens > 0 else 0.0
        nsp_accuracy = total_nsp_correct / total_sequences if total_sequences > 0 else 0.0
        
        metrics = {
            'val_loss': float(val_loss),
            'val_mlm_acc': float(mlm_accuracy),
            'val_nsp_acc': float(nsp_accuracy)
        }
        
        print(f"\nValidation metrics for epoch {epoch}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        
        return metrics 