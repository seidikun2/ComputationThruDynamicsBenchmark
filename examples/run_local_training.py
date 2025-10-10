# -*- coding: utf-8 -*-
# Preview da MazeTask: gera dataset e plota trajetórias 2D + inputs no tempo

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

# --------- eixos de tempo ---------
inputs = to_np(dataset_dict["inputs"])        # (B, T, C)
B, T, C = inputs.shape
if hasattr(env, "t_ms") and len(getattr(env, "t_ms")) == T:
    t = np.asarray(env.t_ms)
else:
    t = np.arange(T)

# --------- canais esperados ---------
has_go      = C >= 3
has_phases  = C >= 7   # init, target, delay, move
has_maze_id = C >= 8

# =========================================================
# 1) TRAJETÓRIAS 2D (alvo no tempo): inputs[...,0:2]
# =========================================================
tx = inputs[..., 0]
ty = inputs[..., 1]

plt.figure(figsize=(6, 5))
for i in range(B):
    plt.plot(tx[i], ty[i], alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.title(f"MazeTask — Trajetórias 2D de alvo (N={B})")
plt.tight_layout()
plt.show()

# Versão colorida por fase (se disponível)
if has_phases:
    phase_names = ["init", "target", "delay", "move"]
    phase_cols  = [3, 4, 5, 6]  # índices nos inputs
    colors      = ["#999999", "#1f77b4", "#ff7f0e", "#2ca02c"]  # cinza, azul, laranja, verde

    # Mostra alguns trials com cor por fase
    n_plot = min(B, 6)
    plt.figure(figsize=(10, 6))
    for i in range(n_plot):
        for ph, col, c in zip(phase_names, phase_cols, colors):
            m = inputs[i, :, col] > 0.5
            # desenha segmentos por fase
            if np.any(m):
                # divide em “runs” contíguos para não ligar segmentos distantes
                idx = np.where(m)[0]
                # separa onde há saltos >1
                splits = np.where(np.diff(idx) > 1)[0] + 1
                chunks = np.split(idx, splits)
                for ch in chunks:
                    if len(ch) > 1:
                        plt.plot(tx[i, ch], ty[i, ch], c=c, lw=2, alpha=0.9)
        # contorno leve com tudo
        plt.plot(tx[i], ty[i], color="k", lw=0.5, alpha=0.3)
    plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
    plt.title("MazeTask — Trajetórias coloridas por fase (amostras)")
    # legenda
    for ph, c in zip(phase_names, colors):
        plt.plot([], [], c=c, lw=3, label=ph)
    plt.legend(frameon=False, ncol=4, fontsize=9)
    plt.tight_layout()
    plt.show()

# =========================================================
# 2) INPUTS NO TEMPO (primeiros N_SHOW_TRIAL trials)
#    Mostra canais: tx,ty, go, fases, maze_id (se existirem)
# =========================================================
def plot_trial_timeseries(i):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, tx[i], label="target_x")
    ax.plot(t, ty[i], label="target_y")

    if has_go:
        ax.plot(t, inputs[i, :, 2], label="go", linestyle="--")

    if has_phases:
        ph_labels = ["phase_init", "phase_target", "phase_delay", "phase_move"]
        for j, name in enumerate(ph_labels, start=3):
            ax.plot(t, inputs[i, :, j], label=name, alpha=0.8)

    if has_maze_id:
        # escalar constante no tempo; normaliza só pra caber no gráfico
        z = inputs[i, :, 7]
        # desenha numa escala separada (ou só mostrar o valor)
        ax.plot(t, z, label=f"maze_id({int(z[0])})", alpha=0.6)

    ax.set_xlabel("time (ms)" if (t.dtype.kind in "iu" or t.dtype.kind == "f") else "time idx")
    ax.set_title(f"Inputs no tempo — trial {i}")
    ax.legend(ncol=3, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()

for i in range(min(B, N_SHOW_TRIAL)):
    plot_trial_timeseries(i)

# =========================================================
# 3) inputs_to_env (se houver; por padrão está vazio)
# =========================================================
if "inputs_to_env" in dataset_dict:
    U = to_np(dataset_dict["inputs_to_env"])  # (B, T, Cenv)
    print("inputs_to_env.shape:", U.shape)
    if U.ndim == 3 and U.shape[2] > 0:
        # overlay
        plt.figure(figsize=(8, 4))
        for i in range(min(B, N_SHOW_TRIAL)):
            for j in range(U.shape[2]):
                plt.plot(t, U[i, :, j], alpha=0.5)
        plt.title("inputs_to_env — trials sobrepostos")
        plt.xlabel("time"); plt.ylabel("amp")
        plt.tight_layout(); plt.show()

        # alguns trials individuais
        for i in range(min(B, 3)):
            plt.figure(figsize=(8, 3.8))
            for j in range(U.shape[2]):
                plt.plot(t, U[i, :, j], label=f"ch {j}")
            plt.title(f"inputs_to_env — trial {i}")
            plt.xlabel("time"); plt.ylabel("amp")
            if U.shape[2] <= 8: plt.legend(ncol=4, fontsize=8, frameon=False)
            plt.tight_layout(); plt.show()
    else:
        print("inputs_to_env sem canais (C=0) — nada a plotar.")
