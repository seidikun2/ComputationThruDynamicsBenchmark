# -*- coding: utf-8 -*-
# Preview da MazeTask: gera dataset e plota
# - Trajetórias 2D coloridas por fase
# - Séries temporais (alvo normalizado, go e fases)
# Fases (sequenciais, sem overlap):
#   INIT  : cursor em repouso, alvo "oculto" (fixo no start)
#   TARGET: alvo revelado, mas ainda sem permissão de mover (go = 0)
#   DELAY : período de espera após ver o alvo (go = 0)
#   MOVE  : liberação do movimento (go = 1)

import os, sys, importlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
N_SAMPLES    = 50
N_TIMESTEPS  = 501   # alinhar com MazeTask
N_SHOW_TRIAL = 6     # quantos trials mostrar individualmente nas séries
MAZE_CFG_DIR = r"C:\Users\User\Documents\GitHub\ComputationThruDynamicsBenchmark\ctd\task_modeling\maze_configs"

# ===== PATHS =====
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
os.environ.setdefault("HOME_DIR", str(REPO_ROOT))

# ===== CARREGA E INSTANCIA MazeTask =====
tenv = importlib.import_module("ctd.task_modeling.task_env.task_env")
MazeTask = getattr(tenv, "MazeTask")
env = MazeTask(export_dir=MAZE_CFG_DIR, n_timesteps=N_TIMESTEPS)

# ===== DATASET =====
dataset_dict, extra_dict = env.generate_dataset(N_SAMPLES)

def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

# --------- inspeciona chaves ---------
print("Chaves no dataset_dict:", list(dataset_dict.keys()))
for k, v in dataset_dict.items():
    try:
        print(f"  {k}: shape={to_np(v).shape}")
    except Exception:
        print(f"  {k}: (sem shape legível)")

# --------- pega tensors úteis ---------
inputs  = to_np(dataset_dict["inputs"])        # (B, T, C)
targets = to_np(dataset_dict["targets"])       # (B, T, 2) – trajetória a ser seguida
B, T, C = inputs.shape

# eixo de tempo (ms, se exportado)
if hasattr(env, "t_ms") and len(getattr(env, "t_ms")) == T:
    t = np.asarray(env.t_ms)
else:
    t = np.arange(T)

# --------- canais esperados ---------
# [0]=tx, [1]=ty, [2]=go, [3]=init, [4]=target, [5]=delay, [6]=move, [7]=maze_id
assert C >= 7, "Esperava ao menos 7 canais (tx,ty,go, init,target,delay,move)."
has_maze_id = C >= 8

# --------- normalização do alvo (para séries comparáveis) ---------
# Normaliza tx/ty pela amplitude global das trajetórias
tx = inputs[..., 0].copy()
ty = inputs[..., 1].copy()
max_abs = max(1e-6, np.nanmax(np.abs(np.stack([tx, ty], axis=-1))))
tx_n = tx / max_abs
ty_n = ty / max_abs
print(f"[normalização] max|target|= {max_abs:.3f}  → tx,ty divididos por esse valor.")

# ========= helper: encontra chunks contíguos de um booleano =========
def bool_runs(mask_1d):
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)

# ========= 1) TRAJETÓRIAS 2D (alvo no tempo), coloridas por fase =========
phase_names = ["init", "target", "delay", "move"]
phase_cols  = [3, 4, 5, 6]  # índices nos inputs
colors      = ["#999999", "#1f77b4", "#ff7f0e", "#2ca02c"]  # cinza, azul, laranja, verde

# Mostra alguns trials com cor por fase sobre o alvo 2D
n_plot = min(B, 12)
plt.figure(figsize=(10, 8))
for i in range(n_plot):
    # contorno leve de toda a trajetória do alvo
    plt.plot(tx[i], ty[i], color="k", lw=0.6, alpha=0.25)

    # pinta trechos por fase
    for ph, col, c in zip(phase_names, phase_cols, colors):
        m = inputs[i, :, col] > 0.5
        for ch in bool_runs(m):
            if len(ch) > 1:
                plt.plot(tx[i, ch], ty[i, ch], c=c, lw=2)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.title("MazeTask — alvo (x,y) colorido por fase")
for ph, c in zip(phase_names, colors):
    plt.plot([], [], c=c, lw=3, label=ph)
plt.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout()
plt.show()

# ========= 2) SÉRIES TEMPORAIS (trial-by-trial) =========
# Para cada trial: alvo normalizado (tx_n, ty_n), go e as quatro fases.
def plot_trial_timeseries(i):
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # séries normalizadas do alvo
    ax.plot(t, tx_n[i], label="target_x (norm.)")
    ax.plot(t, ty_n[i], label="target_y (norm.)")

    # go (0/1)
    ax.plot(t, inputs[i, :, 2], label="go", linestyle="--")

    # marca regiões de fase com faixas translúcidas e linhas base
    for name, col, c in zip(phase_names, phase_cols, colors):
        m = inputs[i, :, col] > 0.5
        # faixa (somente onde há fase)
        for ch in bool_runs(m):
            ax.axvspan(t[ch[0]], t[ch[-1]], color=c, alpha=0.10, linewidth=0)
        # traça a curva 0/1 da fase
        ax.plot(t, inputs[i, :, col], label=f"phase_{name}", color=c, alpha=0.85)

    # maze_id como anotação (se existir)
    if has_maze_id:
        mid = int(inputs[i, 0, 7])
        ax.text(0.02, 0.95, f"maze_id = {mid}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # anotações do que acontece em cada fase (legenda didática)
    # INIT  : alvo oculto (fixo na posição inicial), go=0
    # TARGET: alvo revelado (trajetória aparece), go=0
    # DELAY : espera após ver o alvo (memória do alvo), go=0
    # MOVE  : liberação do movimento, go=1
    txt = ("Fases:\n"
           "INIT  – alvo oculto (fixo no start), go=0\n"
           "TARGET– alvo revelado, ainda sem mover, go=0\n"
           "DELAY – espera após ver o alvo (memória), go=0\n"
           "MOVE  – liberação do movimento, go=1")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=9)

    ax.set_xlabel("tempo (ms)" if (t.dtype.kind in "iu" or t.dtype.kind == "f") else "tempo (índice)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Inputs no tempo — trial {i}")
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

print("\nMostrando séries temporais (alvo normalizado + go + fases) …")
for i in range(min(B, N_SHOW_TRIAL)):
    plot_trial_timeseries(i)

# ========= 3) (Opcional) Comparativo alvo vs. target supervisionado =========
# Mostra que o canal (tx,ty) que a rede recebe corresponde à trajetória alvo esperada.
i = 0
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot(tx[i], ty[i], lw=2)
ax[0].set_title("Alvo (inputs: tx,ty)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].axis("equal")
ax[1].plot(targets[i, :, 0], targets[i, :, 1], lw=2)
ax[1].set_title("Trajetória esperada (targets: x,y)")
ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].axis("equal")
fig.suptitle("Alvo mostrado ao modelo vs. alvo esperado no treino")
fig.tight_layout()
plt.show()
