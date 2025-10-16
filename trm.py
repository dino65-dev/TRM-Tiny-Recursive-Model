import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple
import numpy as np
from rms_norm import RMSNorm
from transformer_block import TransformerBlock

class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model (TRM)
    
    Architecture:
        - Maintains two features: y (answer) and z (latent reasoning)
        - Recursively updates them:
            For n steps: z ← f(x + y + z)  (reasoning with input)
            Then once:   y ← f(y + z)      (update answer without input)
        - Repeat for T cycles
        - Use deep supervision at each cycle
    
    Mathematical formulation:
        Given input x, initialize y=0, z=0
        
        For supervision_step = 1 to N_sup:
            For cycle = 1 to T:
                # Reasoning phase (n steps)
                For i = 1 to n:
                    z ← f_net(x + y + z)
                
                # Prediction phase (1 step)
                y ← f_net(y + z)
            
            Loss += CrossEntropy(y, target)
            
            if should_halt():
                break
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        seq_len: int = 81,  # For Sudoku 9x9
        n_recursions: int = 6,  # n: number of reasoning steps
        t_cycles: int = 3,      # T: number of cycles per supervision
        use_mlp_mixer: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.n_recursions = n_recursions
        self.t_cycles = t_cycles
        
        # Input embedding
        self.embed_input = nn.Embedding(vocab_size, dim)
        
        # Initialize answer and latent embeddings
        # These will be learned starting points
        self.init_y = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.init_z = nn.Parameter(torch.zeros(1, seq_len, dim))
        
        # Single tiny network (shared for all recursions)
        self.network = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                use_mlp_mixer=use_mlp_mixer,
                seq_len=seq_len,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output head: project from embedding to logits
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Adaptive Computation Time (ACT) - predict halt probability
        self.halt_predictor = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        # Xavier/Glorot initialization for embeddings
        nn.init.normal_(self.embed_input.weight, std=0.02)
        nn.init.normal_(self.init_y, std=0.02)
        nn.init.normal_(self.init_z, std=0.02)
        
        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def apply_network(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the tiny network (stack of transformer blocks)
        
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        for block in self.network:
            x = block(x)
        return x
    
    def one_recursion_cycle(
        self, 
        x: torch.Tensor,  # Input embedding
        y: torch.Tensor,  # Current answer
        z: torch.Tensor,  # Latent reasoning
        include_input: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One complete recursion cycle: n updates to z, then 1 update to y
        
        Args:
            x: input embedding (batch, seq_len, dim)
            y: current answer embedding (batch, seq_len, dim)
            z: latent reasoning embedding (batch, seq_len, dim)
            include_input: whether to include x in z updates
            
        Returns:
            y_new: updated answer
            z_new: updated latent
        """
        # Phase 1: Update latent z for n steps (recursive reasoning)
        for _ in range(self.n_recursions):
            if include_input:
                # z ← f(x + y + z)
                z_input = x + y + z
            else:
                # z ← f(y + z) [without input - for inference after training]
                z_input = y + z
            
            z = self.apply_network(z_input)
        
        # Phase 2: Update answer y once (prediction)
        # y ← f(y + z)  [note: no x here, as specified in paper]
        y_input = y + z
        y = self.apply_network(y_input)
        
        return y, z
    
    def forward_with_deep_supervision(
        self,
        x: torch.Tensor,
        num_supervision_steps: int = 16,
        compute_gradients_for_last_only: bool = False
    ) -> Tuple[list, list, list]:
        """
        Forward pass with deep supervision for training.
        
        At each supervision step:
        1. Run T-1 cycles without gradients (detached)
        2. Run 1 final cycle with gradients
        3. Compute loss and accumulate
        
        Args:
            x: input tokens (batch, seq_len) - integers
            num_supervision_steps: K in paper, max refinement steps
            compute_gradients_for_last_only: if True, only last step has grads
            
        Returns:
            all_predictions: list of predictions at each supervision step
            all_halt_probs: list of halt probabilities
            intermediates: list of (y, z) tuples for analysis
        """
        batch_size = x.shape[0]
        
        # Embed input
        x_emb = self.embed_input(x)  # (batch, seq_len, dim)
        
        # Initialize y and z
        y = self.init_y.expand(batch_size, -1, -1).clone()
        z = self.init_z.expand(batch_size, -1, -1).clone()
        
        all_predictions = []
        all_halt_probs = []
        intermediates = []
        
        for sup_step in range(num_supervision_steps):
            # Save current state
            y_checkpoint = y.clone()
            z_checkpoint = z.clone()
            
            # Run T-1 cycles without gradients (efficient inference)
            if self.t_cycles > 1:
                with torch.no_grad():
                    for _ in range(self.t_cycles - 1):
                        y, z = self.one_recursion_cycle(x_emb, y, z)
            
            # Run final cycle with gradients
            if compute_gradients_for_last_only and sup_step < num_supervision_steps - 1:
                with torch.no_grad():
                    y, z = self.one_recursion_cycle(x_emb, y, z)
            else:
                y, z = self.one_recursion_cycle(x_emb, y, z)
            
            # Compute logits from current answer
            logits = self.output_head(y)  # (batch, seq_len, vocab_size)
            
            # Predict halt probability
            # Average across sequence, then predict
            y_pooled = y.mean(dim=1)  # (batch, dim)
            halt_prob = self.halt_predictor(y_pooled).squeeze(-1)  # (batch,)
            
            all_predictions.append(logits)
            all_halt_probs.append(halt_prob)
            intermediates.append((y.detach().clone(), z.detach().clone()))
            
            # Detach for next iteration (breaks gradient flow between supervision steps)
            y = y.detach()
            z = z.detach()
        
        return all_predictions, all_halt_probs, intermediates
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        max_refinement_steps: int = 16,
        halt_threshold: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with adaptive computation time (ACT)
        
        Args:
            x: input tokens (batch, seq_len)
            max_refinement_steps: maximum number of refinement iterations
            halt_threshold: halt when halt_prob > threshold
            
        Returns:
            predictions: (batch, seq_len) - predicted tokens
            exit_steps: (batch,) - which step each example exited at
        """
        self.eval()
        batch_size = x.shape[0]
        
        # Embed input
        x_emb = self.embed_input(x)
        
        # Initialize
        y = self.init_y.expand(batch_size, -1, -1).clone()
        z = self.init_z.expand(batch_size, -1, -1).clone()
        
        # Track which examples have halted
        has_halted = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        exit_steps = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Initialize best predictions tensor (will be populated in first iteration)
        best_predictions = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=x.device)
        
        for step in range(max_refinement_steps):
            # One complete recursion cycle
            for _ in range(self.t_cycles):
                y, z = self.one_recursion_cycle(x_emb, y, z)
            
            # Get current predictions
            logits = self.output_head(y)
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            
            # Compute halt probability
            y_pooled = y.mean(dim=1)
            halt_prob = self.halt_predictor(y_pooled).squeeze(-1)
            
            # Update best predictions for non-halted examples
            mask = ~has_halted
            best_predictions[mask] = preds[mask]
            
            # Check which examples should halt
            should_halt = (halt_prob > halt_threshold) & (~has_halted)
            has_halted = has_halted | should_halt
            exit_steps[should_halt] = step + 1
            
            # If all examples halted, stop
            if has_halted.all():
                break
        
        # For examples that never halted, mark as max steps
        exit_steps[~has_halted] = max_refinement_steps
        
        return best_predictions, exit_steps