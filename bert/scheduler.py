import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        Initialize the scheduler
        
        Args:
            optimizer: The optimizer to wrap (usually Adam)
            d_model: The dimensionality of the model
            n_warmup_steps: Number of warmup steps for the learning rate
        """
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = d_model ** (-0.5)

    def step(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """Get learning rate scale based on warmup steps"""
        return min(self.n_current_steps ** (-0.5), 
                  self.n_current_steps * self.n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """Update learning rate based on current step"""
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Get current learning rate"""
        return self._optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'optimizer': self._optimizer.state_dict(),
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'init_lr': self.init_lr
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.init_lr = state_dict['init_lr'] 