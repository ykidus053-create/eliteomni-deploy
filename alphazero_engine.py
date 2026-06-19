import os, math, random
import numpy as np
from collections import defaultdict

import math
import random
import numpy as np
from collections import defaultdict

# ─── NEURAL NETWORK (policy + value head) ────────────────────────────────────
# Lightweight ResNet-style network — same architecture as AlphaZero
# Policy head: probability over actions
# Value head: scalar win probability [-1, 1]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False

if TORCH:
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(channels)

        def forward(self, x):
            r = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + r)

    class AlphaZeroNet(nn.Module):
        def __init__(self, board_size=6, action_size=36, channels=64, res_blocks=4):
            super().__init__()
            self.board_size  = board_size
            self.action_size = action_size
            # Stem
            self.stem = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            # Residual tower
            self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])
            # Policy head
            self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
            self.policy_bn   = nn.BatchNorm2d(2)
            self.policy_fc   = nn.Linear(2 * board_size * board_size, action_size)
            # Value head
            self.value_conv  = nn.Conv2d(channels, 1, 1, bias=False)
            self.value_bn    = nn.BatchNorm2d(1)
            self.value_fc1   = nn.Linear(board_size * board_size, 64)
            self.value_fc2   = nn.Linear(64, 1)

        def forward(self, x):
            x = self.stem(x)
            x = self.tower(x)
            # Policy
            p = F.relu(self.policy_bn(self.policy_conv(x)))
            p = p.view(p.size(0), -1)
            p = self.policy_fc(p)
            p = F.softmax(p, dim=1)
            # Value
            v = F.relu(self.value_bn(self.value_conv(x)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v))
            return p, v

    class NeuralEvaluator:
        def __init__(self, board_size=6):
            self.net = AlphaZeroNet(board_size=board_size, action_size=board_size*board_size)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
            self.board_size = board_size

        def encode(self, board, player):
            bs = self.board_size
            planes = np.zeros((3, bs, bs), dtype=np.float32)
            for r in range(bs):
                for c in range(bs):
                    if board[r][c] == player:
                        planes[0][r][c] = 1
                    elif board[r][c] != 0:
                        planes[1][r][c] = 1
            planes[2] = player  # whose turn
            return planes

        def predict(self, board, player):
            x = torch.FloatTensor(self.encode(board, player)).unsqueeze(0)
            self.net.eval()
            with torch.no_grad():
                p, v = self.net(x)
            return p.squeeze().numpy(), float(v.squeeze())

        def train_step(self, states, pi_targets, v_targets):
            self.net.train()
            x  = torch.FloatTensor(np.array(states))
            pi = torch.FloatTensor(np.array(pi_targets))
            vt = torch.FloatTensor(np.array(v_targets)).unsqueeze(1)
            p, v = self.net(x)
            loss = F.cross_entropy(p, pi) + F.mse_loss(v, vt)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return float(loss)

        def save(self, path):
            torch.save(self.net.state_dict(), path)

        def load(self, path):
            self.net.load_state_dict(torch.load(path, map_location='cpu'))

else:
    # Fallback: random evaluator if torch not installed
    class NeuralEvaluator:
        def __init__(self, board_size=6):
            self.board_size = board_size
            self.action_size = board_size * board_size

        def predict(self, board, player):
            p = np.ones(self.action_size) / self.action_size
            v = random.uniform(-0.1, 0.1)
            return p, v

        def train_step(self, states, pi_targets, v_targets):
            return 0.0

        def save(self, path): pass
        def load(self, path): pass


# ─── GAME: 6x6 Connect-4 variant ─────────────────────────────────────────────

class MiniGo:
    SIZE = 6

    def __init__(self):
        self.board  = [[0]*self.SIZE for _ in range(self.SIZE)]
        self.player = 1
        self.last   = None
        self.done   = False
        self.winner = 0

    def clone(self):
        g = MiniGo()
        g.board  = [row[:] for row in self.board]
        g.player = self.player
        g.last   = self.last
        g.done   = self.done
        g.winner = self.winner
        return g

    def legal_moves(self):
        if self.done:
            return []
        return [(r, c) for r in range(self.SIZE)
                for c in range(self.SIZE) if self.board[r][c] == 0]

    def play(self, move):
        r, c = move
        self.board[r][c] = self.player
        self.last = move
        if self._check_win(r, c, self.player):
            self.done   = True
            self.winner = self.player
        elif not self.legal_moves():
            self.done   = True
            self.winner = 0
        else:
            self.player = -self.player

    def _check_win(self, r, c, p):
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in dirs:
            cnt = 1
            for sign in (1, -1):
                nr, nc = r + sign*dr, c + sign*dc
                while 0 <= nr < self.SIZE and 0 <= nc < self.SIZE and self.board[nr][nc] == p:
                    cnt += 1
                    nr += sign*dr
                    nc += sign*dc
            if cnt >= 4:
                return True
        return False

    def action_index(self, move):
        return move[0] * self.SIZE + move[1]

    def index_to_move(self, idx):
        return (idx // self.SIZE, idx % self.SIZE)

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("  " + " ".join(str(i) for i in range(self.SIZE)))
        for i, row in enumerate(self.board):
            print(str(i) + " " + " ".join(symbols[c] for c in row))
        print()


# ─── MCTS (UCB + neural prior + value backup) ────────────────────────────────

class MCTSNode:
    __slots__ = ['state','parent','move','children','visits','value_sum','prior','expanded']

    def __init__(self, state, parent=None, move=None, prior=1.0):
        self.state     = state
        self.parent    = parent
        self.move      = move
        self.children  = {}
        self.visits    = 0
        self.value_sum = 0.0
        self.prior     = prior
        self.expanded  = False

    def q(self):
        return self.value_sum / (self.visits + 1e-8)

    def ucb(self, c_puct=1.5):
        if self.parent is None:
            return 0
        return self.q() + c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)


class MCTS:
    def __init__(self, evaluator, simulations=200, c_puct=1.5, temp=1.0):
        self.evaluator   = evaluator
        self.simulations = simulations
        self.c_puct      = c_puct
        self.temp        = temp

    def search(self, game):
        root = MCTSNode(game.clone())
        for _ in range(self.simulations):
            node = root
            # 1. SELECT
            while node.expanded and node.children:
                node = max(node.children.values(), key=lambda n: n.ucb(self.c_puct))
            # 2. EXPAND + EVALUATE
            if not node.state.done:
                policy, value = self.evaluator.predict(node.state.board, node.state.player)
                moves = node.state.legal_moves()
                for move in moves:
                    idx   = node.state.action_index(move)
                    prior = float(policy[idx]) if idx < len(policy) else 1.0/len(moves)
                    child_state = node.state.clone()
                    child_state.play(move)
                    node.children[move] = MCTSNode(child_state, parent=node, move=move, prior=prior)
                node.expanded = True
            else:
                value = node.state.winner * node.state.player
            # 3. BACKUP
            v = value
            while node is not None:
                node.visits    += 1
                node.value_sum += v
                v = -v
                node = node.parent
        # Build policy from visit counts
        visits = {m: c.visits for m, c in root.children.items()}
        total  = sum(visits.values()) + 1e-8
        if self.temp == 0:
            best = max(visits, key=visits.get)
            pi = {m: (1.0 if m == best else 0.0) for m in visits}
        else:
            pi = {m: (v/total)**(1/self.temp) for m, v in visits.items()}
            s  = sum(pi.values())
            pi = {m: v/s for m, v in pi.items()}
        return pi


# ─── SELF-PLAY + TRAINING LOOP ───────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.data   = []
        self.maxlen = maxlen

    def add(self, state_enc, pi, z):
        self.data.append((state_enc, pi, z))
        if len(self.data) > self.maxlen:
            self.data.pop(0)

    def sample(self, n):
        return random.sample(self.data, min(n, len(self.data)))


def self_play_game(mcts, evaluator):
    game    = MiniGo()
    history = []
    while not game.done:
        pi_dict = mcts.search(game)
        moves   = list(pi_dict.keys())
        probs   = [pi_dict[m] for m in moves]
        move    = random.choices(moves, weights=probs)[0]
        # store (encoded_state, full_policy_vector, placeholder_z)
        enc = evaluator.encode(game.board, game.player) if hasattr(evaluator, 'encode') else None
        pi_vec = np.zeros(MiniGo.SIZE * MiniGo.SIZE)
        for m, p in pi_dict.items():
            pi_vec[game.action_index(m)] = p
        history.append((enc, pi_vec, game.player))
        game.play(move)
    # assign outcomes
    winner = game.winner
    samples = []
    for enc, pi_vec, player in history:
        if enc is not None:
            z = 1.0 if winner == player else (-1.0 if winner != 0 else 0.0)
            samples.append((enc, pi_vec, z))
    return samples, game


def train(iterations=5, games_per_iter=10, train_steps=20, simulations=100):
    bs        = MiniGo.SIZE
    evaluator = NeuralEvaluator(board_size=bs)
    mcts      = MCTS(evaluator, simulations=simulations, temp=1.0)
    buffer    = ReplayBuffer()
    model_path = os.path.join(BASE, "alphazero_model.pt")

    print("\nAlphaZero Engine starting...")
    print(f"Board: {bs}x{bs}  |  Simulations/move: {simulations}  |  Torch: {TORCH}\n")

    for it in range(1, iterations + 1):
        print(f"Iteration {it}/{iterations}")
        # Self-play
        wins = {1: 0, -1: 0, 0: 0}
        for g in range(games_per_iter):
            samples, game = self_play_game(mcts, evaluator)
            wins[game.winner] += 1
            for s in samples:
                buffer.add(*s)
        print(f"  Self-play: X={wins[1]} O={wins[-1]} Draw={wins[0]}  Buffer={len(buffer.data)}")
        # Training
        total_loss = 0.0
        for _ in range(train_steps):
            batch = buffer.sample(32)
            if not batch:
                break
            states   = [b[0] for b in batch]
            pi_tgts  = [b[1] for b in batch]
            v_tgts   = [b[2] for b in batch]
            loss     = evaluator.train_step(states, pi_tgts, v_tgts)
            total_loss += loss
        avg_loss = total_loss / max(train_steps, 1)
        print(f"  Avg loss: {avg_loss:.4f}")
        evaluator.save(model_path)
        print(f"  Model saved -> {model_path}")

    print("\nTraining complete. Playing demo game...\n")
    # Demo game: trained agent vs random
    mcts.temp = 0
    game = MiniGo()
    game.render()
    while not game.done:
        if game.player == 1:
            pi = mcts.search(game)
            move = max(pi, key=pi.get)
        else:
            move = random.choice(game.legal_moves())
        game.play(move)
        game.render()
    result = {1: "X wins", -1: "O wins", 0: "Draw"}
    print("Result:", result[game.winner])
    return evaluator

if __name__ == "__main__":
    BASE = os.path.expanduser("~/eliteomni_app")
    train(iterations=5, games_per_iter=10, train_steps=20, simulations=100)
