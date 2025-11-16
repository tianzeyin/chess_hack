from .utils import chess_manager, GameContext
from chess import Move
import chess
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
#                     Global Configuration
# ============================================================

# Download model.pt from Hugging Face (public repo)

# Where to load the model from. We lazy-resolve to avoid blocking import/startup.
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # library not installed or import failed

def _resolve_default_weights() -> str:
    """
    Lazy resolver for model weights path.
    Tries Hugging Face public repo first, then falls back to local file.
    """
    # If user explicitly provided a path (keeps your old behavior intact)
    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        return explicit

    # Try public Hugging Face (no token needed)
    if hf_hub_download is not None:
        try:
            return hf_hub_download(
                repo_id="Miko6836/chesshacks",   # <= your public repo
                filename="model.pt",             # <= your file in the repo
                revision="main"                  # <= branch/tag/commit
            )
        except Exception as e:
            print(f"[WARN] Hugging Face download failed: {e}. Falling back to local file.")

    # Fallback for offline/dev
    return "weights/model.pt"

# NOTE: keep this name so you change as little code as possible.
# IMPORTANT: Use _DEFAULT_WEIGHTS() when you actually load the file.
_DEFAULT_WEIGHTS = _resolve_default_weights  # function, not a path string


# Optional temperature for sampling (0 → argmax).
_TEMPERATURE = float(os.environ.get("AI_TEMPERATURE", "0.0"))

# ============================================================
#                     Move Vocabulary (UCI)
# ============================================================
FILES = "abcdefgh"
RANKS = "12345678"

def _sq_to_idx(file_char: str, rank_char: str) -> int:
    f = FILES.index(file_char)
    r = RANKS.index(rank_char)
    return r * 8 + f

def _idx_to_sq(idx: int) -> str:
    f = idx % 8
    r = idx // 8
    return f"{FILES[f]}{RANKS[r]}"

def _build_move_vocab() -> Tuple[Dict[str, int], List[str]]:
    """
    Build a static vocabulary of all UCI moves across the board:
      - All from!=to pairs (non-promotion)
      - All promotions to q,r,b,n when destination rank is 1 or 8
    """
    uci_list: List[str] = []

    # Base non-promotion moves
    for from_idx in range(64):
        for to_idx in range(64):
            if to_idx == from_idx:
                continue
            uci_list.append(_idx_to_sq(from_idx) + _idx_to_sq(to_idx))

    # Promotion moves (append q/r/b/n) for any move landing on rank 1 or 8
    promos = "qrbn"
    for from_idx in range(64):
        from_sq = _idx_to_sq(from_idx)
        for to_idx in range(64):
            if to_idx == from_idx:
                continue
            to_sq = _idx_to_sq(to_idx)
            if to_sq[1] in ("1", "8"):
                for p in promos:
                    uci_list.append(from_sq + to_sq + p)

    uci_to_idx = {uci: i for i, uci in enumerate(uci_list)}
    return uci_to_idx, uci_list

UCI_TO_IDX, IDX_TO_UCI = _build_move_vocab()
MOVE_SPACE_SIZE = len(IDX_TO_UCI)

# ============================================================
#                     FEN → Tensor Encoding
# ============================================================
# 12 piece planes + 1 side-to-move + 4 castling rights + 1 en-passant file = 18 total
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q':10, 'k':11
}
_NUM_BASE_PLANES = 12

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Encode FEN into a [18, 8, 8] float tensor:
      - 12 planes: piece one-hots
      - 1 plane: side-to-move (ones if white, zeros if black)
      - 4 planes: castling rights (K, Q, k, q)
      - 1 plane: en-passant file (1s down that file)
    """
    parts = fen.split()
    board_part, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
    x = torch.zeros((18, 8, 8), dtype=torch.float32)

    # Pieces
    rows = board_part.split('/')  # rank 8 → 1
    assert len(rows) == 8, f"Bad FEN rows: {board_part}"
    for fen_r, row in enumerate(rows):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                # Our tensor index 0..7 on the rank dimension is from a1 up to h8,
                # So convert FEN's top-down ranks to bottom-up.
                rank_from_bottom = 7 - fen_r
                idx = rank_from_bottom * 8 + file_idx
                r, f = divmod(idx, 8)
                x[PIECE_TO_PLANE[ch], r, f] = 1.0
                file_idx += 1

    # Side to move
    if stm == 'w':
        x[_NUM_BASE_PLANES, :, :] = 1.0

    # Castling rights
    for i, flag in enumerate(['K', 'Q', 'k', 'q']):
        if flag in castling:
            x[_NUM_BASE_PLANES + 1 + i, :, :] = 1.0

    # En-passant file
    ep_plane = _NUM_BASE_PLANES + 1 + 4
    if ep != '-':
        file_char = ep[0]
        if file_char in FILES:
            file_idx = FILES.index(file_char)
            x[ep_plane, :, file_idx] = 1.0

    return x

# ============================================================
#                     Policy Network
# ============================================================
class _ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class ChessPolicyNet(nn.Module):
    def __init__(self, in_channels: int = 18, width: int = 128, n_res: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[_ResidualBlock(width) for _ in range(n_res)])
        self.head = nn.Sequential(
            nn.Conv2d(width, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, MOVE_SPACE_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res(x)
        return self.head(x)  # logits over MOVE_SPACE_SIZE

# ============================================================
#                     Model Loading (Required)
# ============================================================
_model_cache: Optional[ChessPolicyNet] = None
_device: Optional[torch.device] = None

def _load_model_or_die(weights_path: Optional[str] = None) -> ChessPolicyNet:
    """
    Loads the model once and caches it. If weights are missing, raise an error.
    This preserves the ChessHacks requirement: the engine cannot function without the NN.
    """
    global _model_cache, _device
    if _model_cache is not None:
        return _model_cache

    # Lazily resolve weights path
    if weights_path is None:
        # Try HF first (public repo), then fallback to local file
        if hf_hub_download is not None:
            try:
                weights_path = hf_hub_download(
                    repo_id="Miko6836/chesshacks",  # <-- your public repo
                    filename="model.pt",            # <-- your file name in the repo
                    revision="main"                 # <-- branch/tag/commit
                )
            except Exception as e:
                print(f"[WARN] Hugging Face download failed: {e}. Falling back to local file.")
                weights_path = "weights/model.pt"
        else:
            weights_path = "weights/model.pt"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            f"Set MODEL_PATH env var or place weights at weights/model.pt. "
            f"The engine requires the neural network to run."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessPolicyNet().to(_device)

    ckpt = torch.load(weights_path, map_location=_device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    _model_cache = model
    return model

def _clear_model_cache():
    global _model_cache, _device
    _model_cache = None
    _device = None
    # Optional CUDA cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# ============================================================
#                     Inference Helpers
# ============================================================
def _legal_masked_logits(board: chess.Board, logits: torch.Tensor) -> Tuple[List[chess.Move], torch.Tensor]:
    """
    Given a python-chess Board and full logits over global move vocabulary,
    return (legal_moves_list, masked_logits) where masked_logits keeps only legal moves
    (others set to -inf) for proper softmax/argmax.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], logits.new_full(logits.shape, float("-inf"))

    mask = torch.full_like(logits, float("-inf"))
    legal_indices: List[int] = []
    for mv in legal_moves:
        uci = mv.uci()
        # Promotions are lowercase in UCI by convention (python-chess already uses lowercase).
        idx = UCI_TO_IDX.get(uci)
        if idx is not None:
            legal_indices.append(idx)

    if not legal_indices:
        # Fallback: extremely unlikely. Keep engine from crashing.
        return legal_moves, logits.new_full(logits.shape, float("-inf"))

    idx_tensor = torch.tensor(legal_indices, dtype=torch.long, device=logits.device)
    mask[idx_tensor] = logits[idx_tensor]
    return legal_moves, mask

def _choose_move(board: chess.Board, temperature: float = _TEMPERATURE) -> Tuple[chess.Move, Dict[chess.Move, float]]:
    """
    Runs the NN, masks to legal moves, and returns (chosen_move, probabilities_dict).
    """
    # Prepare input tensor from FEN
    fen = board.fen()  # includes side to move, castling, ep, counters
    x = fen_to_tensor(fen).unsqueeze(0)  # [1,18,8,8]

    model = _load_model_or_die()
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)[0]  # [MOVE_SPACE_SIZE]

    # Mask to legal
    legal_moves, masked_logits = _legal_masked_logits(board, logits)

    if not legal_moves:
        # No legal move (checkmate or stalemate).
        return None, {}

    # Convert to probabilities
    if temperature and temperature > 1e-6:
        probs_all = F.softmax(masked_logits / temperature, dim=-1)
    else:
        # Numerical stability: large temperature ~ argmax
        probs_all = torch.zeros_like(masked_logits)
        max_idx = int(torch.argmax(masked_logits).item())
        probs_all[max_idx] = 1.0

    # Build dict for UI logging (only for legal moves)
    probs_dict: Dict[chess.Move, float] = {}
    total_p = 0.0
    for mv in legal_moves:
        idx = UCI_TO_IDX.get(mv.uci())
        if idx is None:
            continue
        p = float(probs_all[idx].item())
        probs_dict[mv] = p
        total_p += p

    # Normalize in case of numerical drift
    if total_p > 0:
        for mv in probs_dict:
            probs_dict[mv] /= total_p
    else:
        # Edge fallback: uniform over legal moves
        uniform_p = 1.0 / len(legal_moves)
        probs_dict = {mv: uniform_p for mv in legal_moves}

    # Sample or argmax according to final probabilities
    if temperature and temperature > 1e-6 and total_p > 0:
        # Sampling
        moves_list = list(probs_dict.keys())
        probs_list = [probs_dict[mv] for mv in moves_list]
        # Turn into torch distribution for sampling
        dist = torch.distributions.Categorical(probs=torch.tensor(probs_list, device=device))
        idx = int(dist.sample().item())
        chosen = moves_list[idx]
    else:
        # Argmax
        chosen = max(probs_dict.items(), key=lambda kv: kv[1])[0]

    return chosen, probs_dict

# ============================================================
#                     ChessHacks Entrypoints
# ============================================================
# Write code here that runs once (at subprocess boot).
# We don't load the model immediately; lazy-load on first move to speed boot & support HMR.

@chess_manager.entrypoint
def test_func(ctx: GameContext) -> Move:
    """
    Called by the devtools backend every time your bot needs to move.
    Must return a python-chess Move that is legal in ctx.board.
    """
    board: chess.Board = ctx.board

    # Make sure we can’t operate without the neural network
    # (satisfies "neural network dependency" rule).
    _ = _load_model_or_die()

    # Run inference
    chosen_move, probs = _choose_move(board, temperature=_TEMPERATURE)

    # Log probabilities (the devtools UI can visualize them)
    # Expectation: dict[Move] -> float (probabilities sum to ~1)
    ctx.logProbabilities(probs if probs is not None else {})

    if chosen_move is None:
        # No legal move available (checkmate/stalemate). Raise to indicate terminal state.
        raise ValueError("No legal moves available.")

    return chosen_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins. Clear any caches/state.
    """
    _clear_model_cache()