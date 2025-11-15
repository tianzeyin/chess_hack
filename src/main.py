from .utils import chess_manager, GameContext
from chess import Move
import torch
import sys
import os
from pathlib import Path

# Import chess_engine from parent directory (same level as src/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_engine import (
    CNNMovePredictor,
    board_to_tensor,
    get_legal_move_mask,
    move_to_index
)

# Load model once at module level
_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _load_model():
    """Load the neural network model."""
    global _model
    if _model is None:
        # Look for model in parent directory (repo root)
        model_path = Path(__file__).parent.parent.parent / "neural_chess_model_final.pth"
        if not model_path.exists():
            # Try in same directory as src/
            model_path = Path(__file__).parent.parent / "neural_chess_model_final.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        _model = CNNMovePredictor()
        _model.load_state_dict(torch.load(model_path, map_location=_device))
        _model.to(_device)
        _model.eval()
        print(f"Loaded model from {model_path} on {_device}")
    
    return _model


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    """
    Get the best move using the neural network.
    This gets called every time the model needs to make a move.
    """
    model = _load_model()
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Convert board to tensor
    board_tensor = board_to_tensor(ctx.board).unsqueeze(0).to(_device)
    
    # Get move predictions from model
    with torch.no_grad():
        move_logits, _ = model(board_tensor)
        move_logits = move_logits[0].cpu().numpy()
    
    # Apply legal move mask
    legal_mask = get_legal_move_mask(ctx.board)
    move_logits = move_logits * legal_mask
    
    # Calculate probabilities for logging
    # Use softmax with temperature for probability distribution
    import numpy as np
    temperature = 1.0
    move_logits_scaled = move_logits / temperature
    
    # Numerical stability: subtract max before exp
    max_logit = np.max(move_logits_scaled)
    exp_logits = np.exp(np.clip(move_logits_scaled - max_logit, -50, 50))
    sum_exp = np.sum(exp_logits)
    
    if sum_exp < 1e-10:
        # Fallback: uniform distribution
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    else:
        probs = exp_logits / sum_exp
        move_probs = {}
        for move in legal_moves:
            move_idx = move_to_index(move)
            if move_idx < len(probs):
                move_probs[move] = float(probs[move_idx])
            else:
                move_probs[move] = 0.0
        
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(move_probs.values())
        if total_prob > 1e-10:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    
    # Log probabilities
    ctx.logProbabilities(move_probs)
    
    # Select best move (highest probability)
    best_move = max(legal_moves, key=lambda m: move_probs.get(m, 0.0))
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Reset function called when a new game begins.
    Can clear caches, reset model state, etc.
    """
    # Model is already loaded and doesn't need reset
    pass
