""" Training with Deep Supervision """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from trm import TinyRecursiveModel

class TRMTrainer:
    """
    Trainer for TRM with deep supervision and ACT
    """
    def __init__(
        self,
        model: TinyRecursiveModel,
        lr: float = 1e-4,
        weight_decay: float = 1.0,
        warmup_steps: int = 2000,
        ema_decay: float = 0.999,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay (important for small datasets)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.base_lr = lr
        self.step_count = 0
        
        # Exponential Moving Average for stability
        self.ema_decay = ema_decay
        self.ema_model = self._create_ema()
        
    def _create_ema(self):
        """Create EMA copy of model"""
        ema = type(self.model)(
            vocab_size=self.model.vocab_size,
            dim=self.model.dim,
            seq_len=self.model.seq_len,
            n_recursions=self.model.n_recursions,
            t_cycles=self.model.t_cycles
        ).to(self.device)
        ema.load_state_dict(self.model.state_dict())
        for param in ema.parameters():
            param.requires_grad = False
        return ema
    
    def _update_ema(self):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )
    
    def _get_lr(self) -> float:
        """Learning rate schedule with warmup"""
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count / self.warmup_steps)
        return self.base_lr
    
    def compute_loss(
        self,
        predictions: list,  # List of logits at each supervision step
        halt_probs: list,   # List of halt probabilities
        targets: torch.Tensor,  # Ground truth tokens
        loss_weights: Optional[list] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss with deep supervision and ACT
        
        Math:
            L_task = sum_k CrossEntropy(pred_k, target)
            L_halt = sum_k BCE(halt_prob_k, is_correct_k)
            L_total = L_task + L_halt
        """
        total_task_loss = torch.tensor(0.0, device=targets.device)
        total_halt_loss = torch.tensor(0.0, device=targets.device)
        num_steps = len(predictions)
        
        # Weight each supervision step (optionally)
        if loss_weights is None:
            loss_weights = [1.0] * num_steps
        
        for step, (logits, halt_prob) in enumerate(zip(predictions, halt_probs)):
            # Task loss: cross-entropy
            # logits: (batch, seq_len, vocab_size)
            # targets: (batch, seq_len)
            batch_size, seq_len, vocab_size = logits.shape
            
            task_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction='mean'
            )
            total_task_loss += task_loss * loss_weights[step]
            
            # ACT halt loss: predict whether current answer is correct
            pred_tokens = logits.argmax(dim=-1)
            is_correct = (pred_tokens == targets).all(dim=1).float()
            
            halt_loss = F.binary_cross_entropy(halt_prob, is_correct)
            total_halt_loss += halt_loss * loss_weights[step]
        
        # Average over supervision steps (use float() to ensure tensor division)
        total_task_loss = total_task_loss / float(num_steps)
        total_halt_loss = total_halt_loss / float(num_steps)
        
        total_loss = total_task_loss + total_halt_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': total_task_loss.item(),
            'halt_loss': total_halt_loss.item()
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        num_supervision_steps: int = 16
    ) -> dict:
        """
        Single training step
        
        Args:
            batch: (inputs, targets)
            num_supervision_steps: number of deep supervision steps
            
        Returns:
            metrics: dictionary of losses and accuracies
        """
        self.model.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass with deep supervision
        predictions, halt_probs, _ = self.model.forward_with_deep_supervision(
            inputs,
            num_supervision_steps=num_supervision_steps
        )
        
        # Compute loss
        loss, metrics = self.compute_loss(predictions, halt_probs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        self.step_count += 1
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
        
        # Update EMA
        self._update_ema()
        
        # Compute accuracy on final prediction
        final_pred = predictions[-1].argmax(dim=-1)
        accuracy = (final_pred == targets).float().mean().item()
        metrics['accuracy'] = accuracy
        metrics['lr'] = self._get_lr()
        
        return metrics