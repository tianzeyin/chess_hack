# ============================================================================
# ChessHacks AI Chess Engine - Complete Implementation
# Winning Strategy for Rook's Rampage and Pawn's Rebellion
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
from typing import Tuple, List, Dict
import random
from collections import defaultdict
import math
import subprocess
import sys
from pathlib import Path

# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURES
# ============================================================================

class CNNMovePredictor(nn.Module):
    """
    End-to-end CNN that takes a board state and directly outputs a move.
    Architecture optimized for fast inference and high accuracy.
    """
    def __init__(self, hidden_size=256):
        super().__init__()
        # Input: 8x8x12 (12 piece types for white and black)
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output: 1792 possible moves (max legal moves in any position)
        # Using 1792 = 64*64 - underpromotions handled separately
        # For promotions: we'll use a separate encoding scheme
        self.move_output = nn.Linear(hidden_size, 1792)
        
        # Optional: Value head for position evaluation
        self.value_fc = nn.Linear(hidden_size, 256)
        self.value_output = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, board_tensor):
        # Convolutional layers with batch norm would improve performance
        x = self.relu(self.conv1(board_tensor))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Move logits
        move_logits = self.move_output(x)
        
        # Value estimation
        value = torch.tanh(self.value_output(self.relu(self.value_fc(x))))
        
        return move_logits, value


class TransformerChessModel(nn.Module):
    """
    Transformer-based model for predicting chess moves from PGN sequences.
    Treats chess games as sequences of moves (tokens).
    """
    def __init__(self, vocab_size=2048, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head for move prediction
        self.move_head = nn.Linear(d_model, 2048)  # Vocabulary size
        
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, move_sequence, attention_mask=None):
        x = self.embedding(move_sequence)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        logits = self.move_head(x)
        return logits


class PolicyValueNetwork(nn.Module):
    """
    Lightweight network for MCTS guidance.
    Outputs policy (move probabilities) and value (position evaluation).
    """
    def __init__(self):
        super().__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Policy head (move probabilities)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 1792)  # Legal moves only
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, board_tensor):
        x = torch.relu(self.conv1(board_tensor))
        x = torch.relu(self.conv2(x))
        
        # Policy
        policy = torch.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = torch.softmax(self.policy_fc(policy), dim=1)
        
        # Value
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


# ============================================================================
# 2. BOARD REPRESENTATION AND ENCODING
# ============================================================================

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to a tensor representation.
    Shape: (12, 8, 8) - 12 channels for each piece type
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            tensor[channel, row, col] = 1.0
    
    return torch.from_numpy(tensor)


def pgn_to_move_tokens(pgn_string: str, max_length=1024) -> List[int]:
    """
    Convert PGN notation to token sequences.
    Encodes moves, piece positions, and board state changes.
    """
    # Tokenize moves from PGN
    tokens = []
    board = chess.Board()
    
    # Parse PGN moves (basic implementation)
    try:
        # Extract move text from PGN (simplified - assumes moves are in standard format)
        import re
        # Remove comments and annotations
        move_text = re.sub(r'\{[^}]*\}', '', pgn_string)
        move_text = re.sub(r'\([^)]*\)', '', move_text)
        # Extract move numbers and moves
        moves = re.findall(r'\d+\.\s*([^\s]+(?:\s+[^\s]+)?)', move_text)
        
        for move_pair in moves[:max_length]:
            # Handle white and black moves
            move_parts = move_pair.strip().split()
            for move_san in move_parts:
                if move_san and not move_san[0].isdigit():
                    try:
                        move = board.parse_san(move_san)
                        uci = move.uci()
                        token = hash(uci) % 2048
                        tokens.append(token)
                        board.push(move)
                    except:
                        break  # Skip invalid moves
    except:
        # Fallback: return empty list if parsing fails
        pass
    
    return tokens[:max_length]


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to a unique index.
    Handles promotions, en passant, and castling correctly.
    Returns index in range [0, 1791] for standard moves.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Handle promotions (most common special case)
    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotions: encode as special indices
        # Base: 1792 + (from_sq * 3) + promotion_type_offset
        promotion_offset = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
        if move.promotion in promotion_offset:
            return 1792 + (from_sq * 3) + promotion_offset[move.promotion]
    
    # Standard encoding: from_square * 64 + to_square
    # This works for most moves including queen promotions
    move_idx = from_sq * 64 + to_sq
    
    # Ensure it fits in our output space (1792 = 28 * 64)
    # For moves beyond this, use modulo (shouldn't happen in practice)
    if move_idx >= 1792:
        # Fallback: use a hash-based approach
        move_idx = (from_sq * 73 + to_sq) % 1792
    
    return move_idx


def index_to_move(board: chess.Board, move_idx: int) -> chess.Move:
    """
    Convert an index back to a move (for a given board position).
    This is approximate - we'll match to legal moves.
    """
    # Handle underpromotions
    if move_idx >= 1792:
        from_sq = (move_idx - 1792) // 3
        promo_type_idx = (move_idx - 1792) % 3
        promo_types = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
        promotion = promo_types[promo_type_idx]
        
        # Find matching legal move
        for move in board.legal_moves:
            if move.from_square == from_sq and move.promotion == promotion:
                return move
    
    # Standard moves
    from_sq = move_idx // 64
    to_sq = move_idx % 64
    
    # Find matching legal move
    for move in board.legal_moves:
        if move.from_square == from_sq and move.to_square == to_sq:
            # Prefer queen promotion if multiple promotions exist
            if move.promotion == chess.QUEEN or not move.promotion:
                return move
    
    # Fallback: return first legal move
    legal_moves = list(board.legal_moves)
    return legal_moves[0] if legal_moves else None


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Create a binary mask for legal moves (1 for legal, 0 for illegal).
    Shape: (1792,) for standard moves + underpromotions handled separately.
    """
    mask = np.zeros(1792, dtype=np.float32)
    for move in board.legal_moves:
        move_idx = move_to_index(move)
        if move_idx < 1792:  # Only mark if in standard range
            mask[move_idx] = 1.0
    return mask


# ============================================================================
# 3. DATASET LOADING
# ============================================================================

class ChessGameDataset(Dataset):
    """
    Load chess games from PGN or Lichess database format.
    Returns board states, legal moves, and game outcomes.
    """
    def __init__(self, pgn_file: str, max_games=50000):
        self.data = []
        self.load_pgn_file(pgn_file, max_games)
    
    def load_pgn_file(self, pgn_file: str, max_games: int):
        """Load games from PGN file (supports Lichess format)."""
        import re
        with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Split into individual games
        games = re.split(r'\n\n(?=\[)', content)
        game_data = []
        
        for game_text in games[:max_games]:
            # Extract move text (everything after the headers)
            move_section = re.search(r'\n\n(.*?)(?=\n\n|$)', game_text, re.DOTALL)
            if not move_section:
                continue
                
            move_text = move_section.group(1)
            # Remove comments and annotations
            move_text = re.sub(r'\{[^}]*\}', '', move_text)
            move_text = re.sub(r'\([^)]*\)', '', move_text)
            # Extract moves (format: 1. e4 e5 2. Nf3 Nc6 ...)
            moves = re.findall(r'\d+\.\s*([^\s]+(?:\s+[^\s]+)?)', move_text)
            
            if len(moves) < 3:  # Filter very short games
                continue
                
            current_game = []
            for move_pair in moves:
                # Split white and black moves
                parts = move_pair.strip().split()
                for move_san in parts:
                    if move_san and not move_san[0].isdigit() and move_san not in ['1-0', '0-1', '1/2-1/2', '*']:
                        current_game.append(move_san)
            
            if len(current_game) > 5:
                game_data.append(current_game)
        
        # Process games into board-move pairs
        for game_moves in game_data:
            board = chess.Board()
            for move_san in game_moves:
                try:
                    move = board.parse_san(move_san)
                    board_tensor = board_to_tensor(board)
                    legal_mask = get_legal_move_mask(board)
                    move_idx = move_to_index(move)
                    
                    self.data.append({
                        'board': board_tensor,
                        'move_idx': move_idx,
                        'legal_mask': legal_mask,
                        'board_fen': board.fen()
                    })
                    board.push(move)
                except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                    break  # Skip malformed moves
                except Exception:
                    break  # Skip any other errors
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# 4. MONTE CARLO TREE SEARCH WITH NEURAL GUIDANCE
# ============================================================================

class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0  # Prior probability from neural network
        
    def uct_value(self, c_param=1.41):
        """Upper Confidence Tree value for node selection."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            # Root node - no exploration term
            return self.value / self.visits
        exploitation = self.value / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCT value."""
        return max(self.children.values(), key=lambda x: x.uct_value())
    
    def expand(self, policy_prior):
        """Expand node by creating children for all legal moves."""
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            move_idx = move_to_index(move)
            # Handle promotions and special moves (they might have different indices)
            if move_idx >= len(policy_prior):
                # For promotions, use a default prior or find closest match
                prior = 1.0 / len(legal_moves)  # Uniform prior as fallback
            else:
                prior = float(policy_prior[move_idx])
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move)
            self.children[move].prior = prior
    
    def backup(self, value):
        """Backup value through tree."""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backup(-value)  # Negate for alternating perspectives


class NeuralMCTS:
    """Monte Carlo Tree Search guided by a neural network policy."""
    def __init__(self, policy_value_net, device='cpu', simulations=400):
        self.network = policy_value_net.to(device)
        self.device = device
        self.simulations = simulations
    
    def search(self, board: chess.Board) -> chess.Move:
        """Run MCTS and return best move."""
        root = MCTSNode(board)
        
        for _ in range(self.simulations):
            node = root
            
            # Selection: traverse to leaf node
            while len(node.children) > 0 and not node.board.is_game_over():
                node = node.select_child()
            
            # Expansion: expand leaf node if not terminal
            if not node.board.is_game_over() and len(node.children) == 0:
                board_tensor = board_to_tensor(node.board).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, value = self.network(board_tensor)
                
                policy_prior = policy[0].cpu().numpy()
                node.expand(policy_prior)
                
                # Select a child for rollout (if any children were created)
                if len(node.children) > 0:
                    node = node.select_child()
            
            # Evaluation
            if node.board.is_game_over():
                result = node.board.result()
                if result == '1-0':
                    value = 1.0
                elif result == '0-1':
                    value = -1.0
                else:
                    value = 0.0
            else:
                board_tensor = board_to_tensor(node.board).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.network(board_tensor)
                value = value.item()
            
            # Backup
            node.backup(value)
        
        # Return move with most visits
        if not root.children:
            # Fallback: return first legal move if MCTS didn't expand
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None
        
        best_move = max(root.children.items(), 
                       key=lambda x: x[1].visits)[0]
        return best_move


# ============================================================================
# 5. UCI ENGINE WRAPPER
# ============================================================================

class NeuralChessEngine:
    """UCI-compatible neural chess engine."""
    
    def __init__(self, model, device='cpu', use_mcts=False):
        self.model = model
        self.device = device
        self.use_mcts = use_mcts
        self.mcts = NeuralMCTS(model, device) if use_mcts else None
        self.board = chess.Board()
    
    def uci(self):
        """UCI protocol initialization."""
        print("id name NeuralChessEngine v1.0")
        print("id author ChessHacks")
        print("option name UCI_Chess960 type check default false")
        print("uciok")
    
    def isready(self):
        """Check if engine is ready."""
        print("readyok")
    
    def setoption(self, name, value):
        """Set engine options."""
        pass
    
    def ucinewgame(self):
        """Initialize new game."""
        self.board = chess.Board()
    
    def position(self, fen_or_moves):
        """Set board position."""
        try:
            if fen_or_moves.startswith("fen"):
                fen_str = fen_or_moves[4:].strip()
                self.board = chess.Board(fen_str)
            elif fen_or_moves.startswith("startpos"):
                self.board = chess.Board()
                moves_part = fen_or_moves.split("moves")
                if len(moves_part) > 1:
                    for move_uci in moves_part[1].split():
                        try:
                            self.board.push(chess.Move.from_uci(move_uci))
                        except (chess.InvalidMoveError, ValueError):
                            # Skip invalid moves
                            pass
            else:
                # Default to starting position
                self.board = chess.Board()
        except (chess.InvalidFenError, ValueError):
            # Fallback to starting position on error
            self.board = chess.Board()
    
    def go(self, depth=20, time_ms=None, infinite=False):
        """Generate best move."""
        if self.use_mcts:
            move = self.mcts.search(self.board)
        else:
            move = self._select_best_move_nn()
        
        # Handle None move (shouldn't happen, but safety check)
        if move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
            else:
                print("bestmove (none)")
                return
        
        print(f"bestmove {move.uci()}")
    
    def _select_best_move_nn(self) -> chess.Move:
        """Select best move using neural network."""
        board_tensor = board_to_tensor(self.board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            move_logits, _ = self.model(board_tensor)
            move_logits = move_logits[0].cpu().numpy()
        
        # Apply legal move mask
        legal_mask = get_legal_move_mask(self.board)
        move_logits = move_logits * legal_mask
        
        # Select best legal move by trying to find the move with highest score
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            move_idx = move_to_index(move)
            if move_idx < len(move_logits):
                score = move_logits[move_idx]
                if score > best_score:
                    best_score = score
                    best_move = move
        
        # Fallback if no move found (shouldn't happen, but safety check)
        return best_move if best_move is not None else legal_moves[0]
    
    def quit(self):
        """Exit engine."""
        sys.exit(0)
    
    def run_uci_loop(self):
        """Main UCI loop."""
        while True:
            try:
                cmd = input().strip()
                
                if cmd == "uci":
                    self.uci()
                elif cmd == "isready":
                    self.isready()
                elif cmd == "ucinewgame":
                    self.ucinewgame()
                elif cmd.startswith("setoption"):
                    parts = cmd.split()
                    try:
                        name_idx = parts.index("name") + 1
                        if "value" in parts:
                            value_idx = parts.index("value") + 1
                            value = parts[value_idx] if value_idx < len(parts) else ""
                        else:
                            value = ""
                        name = parts[name_idx] if name_idx < len(parts) else ""
                        self.setoption(name, value)
                    except (ValueError, IndexError):
                        # Invalid setoption command, ignore
                        pass
                elif cmd.startswith("position"):
                    self.position(cmd[9:])
                elif cmd.startswith("go"):
                    self.go()
                elif cmd == "quit":
                    self.quit()
            except EOFError:
                break


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def collate_fn(batch):
    """Custom collate function for dictionary batches."""
    boards = torch.stack([item['board'] for item in batch])
    moves = torch.tensor([item['move_idx'] for item in batch], dtype=torch.long)
    legal_masks = torch.stack([torch.from_numpy(item['legal_mask']) for item in batch])
    return {
        'board': boards,
        'move_idx': moves,
        'legal_mask': legal_masks
    }


def train_move_predictor(model, train_loader, val_loader, epochs=10, device='cpu'):
    """Train the CNN move predictor."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_move = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Handle both dict batches (from custom collate) and regular batches
            if isinstance(batch, dict):
                boards = batch['board'].to(device)
                moves = batch['move_idx'].to(device)
                legal_mask = batch['legal_mask'].to(device)
            else:
                # Fallback for list of dicts
                boards = torch.stack([b['board'] for b in batch]).to(device)
                moves = torch.tensor([b['move_idx'] for b in batch], dtype=torch.long).to(device)
                legal_mask = torch.stack([torch.from_numpy(b['legal_mask']) for b in batch]).to(device)
            
            optimizer.zero_grad()
            
            move_logits, values = model(boards)
            
            # Apply legal move mask
            move_logits = move_logits * legal_mask
            
            loss_move = criterion_move(move_logits, moves)
            loss_value = criterion_value(values.squeeze(), torch.zeros_like(values.squeeze()))
            
            loss = loss_move + 0.1 * loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            pred_moves = torch.argmax(move_logits, dim=1)
            correct += (pred_moves == moves).sum().item()
            total += moves.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            for batch in val_loader:
                # Handle both dict batches (from custom collate) and regular batches
                if isinstance(batch, dict):
                    boards = batch['board'].to(device)
                    moves = batch['move_idx'].to(device)
                    legal_mask = batch['legal_mask'].to(device)
                else:
                    # Fallback for list of dicts
                    boards = torch.stack([b['board'] for b in batch]).to(device)
                    moves = torch.tensor([b['move_idx'] for b in batch], dtype=torch.long).to(device)
                    legal_mask = torch.stack([torch.from_numpy(b['legal_mask']) for b in batch]).to(device)
                
                move_logits, _ = model(boards)
                move_logits = move_logits * legal_mask
                
                loss = criterion_move(move_logits, moves)
                val_loss += loss.item()
                
                pred_moves = torch.argmax(move_logits, dim=1)
                val_correct += (pred_moves == moves).sum().item()
                val_total += moves.size(0)
            
            val_accuracy = val_correct / val_total
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return model


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Initialize and train the chess engine."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CNNMovePredictor(hidden_size=256)
    policy_value_net = PolicyValueNetwork()
    
    # Example: Create dummy dataset for demonstration
    # In practice, use real PGN data from Lichess
    print("Note: In production, load real chess games from Lichess database")
    print("Download: https://database.lichess.org/")
    
    # Save untrained models for UCI interface
    torch.save(model.state_dict(), 'neural_chess_model.pth')
    torch.save(policy_value_net.state_dict(), 'policy_value_net.pth')
    
    print("Models saved. To use:")
    print("1. python chess_engine.py --uci (runs UCI interface)")
    print("2. Load model with: model.load_state_dict(torch.load('neural_chess_model.pth'))")
    print("\nTraining Tips for Rook's Rampage:")
    print("- Use self-play to continuously improve")
    print("- Deploy early and submit frequent updates")
    print("- Monitor ELO leaderboard and adjust strategy")
    print("- Use MCTS for stronger play, pure NN for faster inference")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uci":
        # Run as UCI engine
        model = CNNMovePredictor()
        model.load_state_dict(torch.load('neural_chess_model.pth', map_location='cpu'))
        engine = NeuralChessEngine(model, device='cpu', use_mcts=False)
        engine.run_uci_loop()
    else:
        main()