# Class to generate training data for task-trained RNN that does 3 bit memory task
import json
import torch
import torch.nn                             as nn
import gymnasium                            as gym
import matplotlib.pyplot                    as plt
import numpy                                as np
from   abc                                  import ABC, abstractmethod
from   typing                               import Any, Optional, Union, Sequence
from   pathlib                              import Path
from   gymnasium                            import spaces
from   motornet.environment                 import Environment
from   numpy                                import ndarray
from   torch._tensor                        import Tensor
from   ctd.task_modeling.task_env.loss_func import NBFFLoss, RandomTargetLoss

class DecoupledEnvironment(gym.Env, ABC):
    """
    Abstract class representing a decoupled environment.
    This class is abstract and cannot be instantiated.

    """

    # All decoupled environments should have
    # a number of timesteps and a noise parameter

    @abstractmethod
    def __init__(self, n_timesteps: int, noise: float):
        super().__init__()
        self.dataset_name = "DecoupledEnvironment"
        self.n_timesteps  = n_timesteps
        self.noise        = noise

    # All decoupled environments should have
    # functions to reset, step, and generate trials
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_dataset(self, n_samples):
        """Must return a dictionary with the following keys:
        #----------Mandatory keys----------
        ics:          initial conditions
        inputs:       inputs to the environment
        targets:      targets of the environment
        conds:        conditions information (if applicable)
        extra:        extra information
        #----------Optional keys----------
        true_inputs:  true inputs to the environment (if applicable)
        true_targets: true targets of the environment (if applicable)
        phase_dict:   phase information (if applicable)
        """
        pass

class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy environment where the goal is to flip the required bit.
    """

    def __init__(self,  n_timesteps: int,   noise: float,     n=1,
        switch_prob=0.01, transition_blind=4,dynamic_noise=0,):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name      = f"{n}BFF"
        self.action_space      = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(n,), dtype=np.float32)
        self.context_inputs    = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)
        self.n                 = n
        self.state             = np.zeros(n)
        self.input_labels      = [f"Input {i}" for i in range(n)]
        self.output_labels     = [f"Output {i}" for i in range(n)]
        self.noise             = noise
        self.dynamic_noise     = dynamic_noise
        self.coupled_env       = False
        self.switch_prob       = switch_prob
        self.transition_blind  = transition_blind
        self.loss_func         = NBFFLoss(transition_blind=transition_blind)

    def step(self, action):
        # Generates state update given an input to the flip-flop
        for i in range(self.n):
            if action[i] == 1:
                self.state[i]  = 1
            elif action[i] == -1:
                self.state[i]  = 0

    def generate_trial(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand                                  = np.random.random(size=(self.n_timesteps, self.n))
        inputs                                     = np.zeros((self.n_timesteps, self.n))
        inputs[inputRand > (1 - self.switch_prob)] = 1  # 2% chance of flipping up or down
        inputs[inputRand <(self.switch_prob)]      = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :]                             = 0

        # Generate the desired outputs given the inputs
        outputs                                    = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :]                          = self.state

        # Add noise to the inputs for the trial
        true_inputs    = inputs
        inputs         = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def reset(self):
        self.state     = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        n_timesteps    = self.n_timesteps
        ics_ds         = np.zeros(shape=(n_samples, self.n))
        outputs_ds     = np.zeros(shape=(n_samples, n_timesteps, self.n))
        inputs_ds      = np.zeros(shape=(n_samples, n_timesteps, self.n))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i,:,:]            = outputs
            inputs_ds[i,:,:]             = inputs
            true_inputs_ds[i,:,:]        = true_inputs

        dataset_dict = {
            "ics":           ics_ds,
            "inputs":        inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets":       outputs_ds,
            "true_inputs":   true_inputs_ds,
            "conds":         np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra":         np.zeros(shape=(n_samples, 1)),
        }
        extra_dict        = {}
        return dataset_dict, extra_dict

    def render(self):
        inputs, states, _ = self.generate_trial()
        fig1, axes        = plt.subplots(nrows=self.n + 1, ncols=1, sharex=True)
        colors            = plt.cm.rainbow(np.linspace(0, 1, self.n))
        for i in range(self.n):
            axes[i].plot(states[:, i], color=colors[i])
            axes[i].set_ylabel(f"State {i}")
            axes[i].set_ylim(-0.2, 1.2)
        ax2               = axes[-1]
        for i in range(self.n):
            ax2.plot(inputs[:, i], color=colors[i])
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Inputs")
        plt.tight_layout()
        plt.show()
        fig1.savefig("nbitflipflop.pdf")

    def render_3d(self, n_trials=10):
        if self.n > 2:
            fig           = plt.figure(figsize=(5 * n_trials, 5))
            # Make colormap for the timestep in a trial
            for i in range(n_trials):
                ax        = fig.add_subplot(1, n_trials, i + 1, projection="3d")
                inputs, states, _ = self.generate_trial()
                ax.plot(states[:, 0], states[:, 1], states[:, 2])
                ax.set_xlabel("Bit 1")
                ax.set_ylabel("Bit 2")
                ax.set_zlabel("Bit 3")
                ax.set_title(f"Trial {i+1}")
            plt.tight_layout()
            plt.show()

class MazeTask:
    """Ambiente 2D com paredes retangulares (AABB) e alvo XY."""

    def __init__(self, 
                 export_dir: Union[str, Path], 
                 n_timesteps: int, 
                 dt: float = 0.01, 
                 speed_limit: float = 100.0, 
                 agent_radius: float = 0.0, 
                 t_ms: Sequence[int] = np.arange(-50, 451),  # grade desejada, em ms
                 use_maze_ids: Optional[Sequence[int]] = None,  
                 use_versions: Optional[Sequence[int]] = None,
                 go_cue_at_zero: bool = True, 
                 add_xy_noise: float = 0.0,):
        self.dataset_name       = "MazeWorld2D"
        self.n_timesteps        = int(n_timesteps)
        self.dt                 = float(dt)
        self.speed_limit        = float(speed_limit)
        self.agent_radius       = float(agent_radius)
        self.coupled_env        = True
        self.state_label        = "xy"
        self.dynamic_noise      = 0.0

        # labels para o DataModule
        self.input_labels       = ["tx_norm", "ty_norm", "go", "phase_init", "phase_target",
                                   "phase_delay", "phase_move", "maze_id"]
        self.output_labels      = ["vx", "vy"]  # ação do controlador (velocidade XY)

        # spaces de observação/ação/contexto
        self.observation_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.context_inputs     = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

        # loss compatível com o wrapper (usa dicionário com "controlled" vs "targets")
        self.loss_func          = RandomTargetLoss(position_loss=nn.MSELoss(), pos_weight=1.0, act_weight=0.0)

        export_dir              = Path(export_dir)
        with open(export_dir / "mazes.json", "r", encoding="utf-8") as f:
            self.mazes          = json.load(f)
        self.traj_npz           = np.load(export_dir / "traj_means.npz")

        self.use_maze_ids       = set(use_maze_ids) if use_maze_ids is not None else None
        self.use_versions       = set(use_versions) if use_versions is not None else None
        self.keys               = self._collect_valid_keys()  # lista de strings "maze{mid}_ver{ver}"
        any_k                   = next(iter(self.keys))
        self.t_ms_export        = self.traj_npz[f"{any_k}_t_ms"].astype(int)
        self.t_ms               = np.asarray(t_ms, dtype=int)
        self.go_cue_at_zero     = bool(go_cue_at_zero)
        self.add_xy_noise       = float(add_xy_noise)

        # estado interno (tensors quando em rollout)
        self._pos_t             = None    # (B,2) tensor
        self._goal_t            = None    # (B,2) tensor
        self._rects_t           = None    # (R,4) tensor [xmin,xmax,ymin,ymax]
        self._cur_key           = None    # "maze{mid}_ver{ver}"
        self._terminated        = False

        # se n_timesteps não bate com len(t_ms), force T = len(t_ms)
        if self.n_timesteps    != len(self.t_ms):
            self.n_timesteps    = len(self.t_ms)

    # ---------- utils geométricos ----------

    @staticmethod
    def _clamp_norm_np(v: np.ndarray, maxnorm: float) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
        f = np.minimum(1.0, maxnorm / n)
        return v * f

    @staticmethod
    def _clamp_norm_torch(v: torch.Tensor, maxnorm: float) -> torch.Tensor:
        eps   = 1e-9
        n     = torch.linalg.norm(v, dim=-1, keepdim=True) + eps
        scale = torch.clamp(torch.as_tensor(maxnorm, dtype=v.dtype, device=v.device) / n, max=1.0)
        return v * scale

    @staticmethod
    def _rects_from_json(maze_dict):
        rects = []
        for b in maze_dict.get("barriers", []):
            if isinstance(b, dict):
                cx, cy         = float(b.get("cx", 0)), float(b.get("cy", 0))
                hw, hh         = float(b.get("hw", 0)), float(b.get("hh", 0))
            elif isinstance(b, (list, tuple)) and len(b) >= 4:
                cx, cy, hw, hh = map(float, b[:4])
            else:
                continue
            rects.append([cx - hw, cx + hw, cy - hh, cy + hh])
        return rects

    def _collect_valid_keys(self):
        keys                  = []
        for k in self.mazes.keys():  # formato "mazeId__version"
            mid_str, ver_str  = k.split("__")
            mid, ver          = int(mid_str), int(ver_str)
            if (self.use_maze_ids is not None) and (mid not in self.use_maze_ids):
                continue
            if (self.use_versions is not None) and (ver not in self.use_versions):
                continue
            base              = f"maze{mid}_ver{ver}"
            if (f"{base}_X" in self.traj_npz) and (f"{base}_Y" in self.traj_npz) and (f"{base}_t_ms" in self.traj_npz):
                keys.append(base)
        if not keys:
            raise RuntimeError("Sem chaves válidas no NPZ para os filtros.")
        return keys

    def _sample_key(self, rng):
        key                   = rng.choice(self.keys)
        mid                   = int(key.split("_")[0].replace("maze", ""))
        ver                   = int(key.split("_")[1].replace("ver", ""))
        maze_json_key         = f"{mid}__{ver}"
        return key, self.mazes[maze_json_key]

    # ---------- alinhamento de trajetórias ----------
    def _align_traj(self, X_src, Y_src, t_src, t_dst):
        """Retorna X,Y interpolados em t_dst (mesma length de t_dst)."""
        X_src                 = np.asarray(X_src, dtype=np.float32)
        Y_src                 = np.asarray(Y_src, dtype=np.float32)
        t_src                 = np.asarray(t_src, dtype=np.float32)
        t_dst                 = np.asarray(t_dst, dtype=np.float32)
        x_al                  = np.interp(t_dst, t_src, X_src, left=X_src[0], right=X_src[-1]).astype(np.float32)
        y_al                  = np.interp(t_dst, t_src, Y_src, left=Y_src[0], right=Y_src[-1]).astype(np.float32)
        return x_al, y_al

    # ---------- API esperada pelo wrapper ----------
    def reset(self, batch_size: int = 1, options: Optional[dict] = None):
        rng                   = np.random.default_rng()
        self._terminated      = False

        # device alvo (se passado via wrapper)
        target_device         = None
        if options is not None:
            target_device     = options.get("device", None)

        # escolhe UM maze/versão e guarda para o episódio
        key, maze_dict        = self._sample_key(rng)
        self._cur_key         = key
        rects_np              = self._rects_from_json(maze_dict)

        X                     = self.traj_npz[f"{key}_X"].astype(np.float32)
        Y                     = self.traj_npz[f"{key}_Y"].astype(np.float32)
        t                     = self.traj_npz[f"{key}_t_ms"].astype(np.float32)
        Xr, Yr                = self._align_traj(X, Y, t_src=t, t_dst=self.t_ms)

        p0_np                 = np.array([Xr[0], Yr[0]], dtype=np.float32)

        # estado interno como tensor (no device certo)
        dev                   = torch.device(target_device) if target_device is not None else torch.device("cpu")
        self._pos_t           = torch.as_tensor(p0_np, dtype=torch.float32, device=dev).expand(batch_size, 2).clone()
        self._goal_t          = torch.as_tensor([Xr[-1], Yr[-1]], dtype=torch.float32, device=dev).expand(batch_size, 2).clone()
        if len(rects_np) > 0:
            self._rects_t     = torch.as_tensor(rects_np, dtype=torch.float32, device=dev)  # (R,4)
        else:
            self._rects_t     = None

        # saída
        obs                   = self._pos_t
        joint_state           = torch.zeros((obs.shape[0], 0), dtype=obs.dtype, device=obs.device)  # (B,0)
        info                  = {"states": {"xy": obs, "joint": joint_state}}
        return obs, info

    def _resolve_collision_torch(self, pos: torch.Tensor) -> torch.Tensor:
        """Empurra pontos (B,2) para fora de retângulos (R,4) usando menor penetração."""
        if self._rects_t is None:
            return pos

        # (B,2) -> (B,1) broadcast
        x, y                 = pos[:, 0].unsqueeze(1), pos[:, 1].unsqueeze(1)   # (B,1)
        xmin, xmax           = self._rects_t[:, 0], self._rects_t[:, 1]         # (R,)
        ymin, ymax           = self._rects_t[:, 2], self._rects_t[:, 3]         # (R,)
        ar                   = torch.as_tensor(self.agent_radius, dtype=pos.dtype, device=pos.device)

        inside               = (x >= (xmin - ar)) & (x <= (xmax + ar)) & (y >= (ymin - ar)) & (y <= (ymax + ar))  # (B,R)
        if not inside.any():
            return pos

        dx_left              = (x - (xmin - ar)).abs()               # (B,R)
        dx_right             = ((xmax + ar) - x).abs()
        dy_bot               = (y - (ymin - ar)).abs()
        dy_top               = ((ymax + ar) - y).abs()

        d                    = torch.stack([dx_left, dx_right, dy_bot, dy_top], dim=-1)  # (B,R,4)
        # invalida distâncias de retângulos nos quais o ponto não está dentro
        d[~inside.unsqueeze(-1).expand_as(d)] = float("inf")

        # argmin em (R,4) por batch
        B, R, _              = d.shape
        flat_idx             = d.view(B, -1).argmin(dim=1)        # (B,)
        which_r              = flat_idx // 4                      # (B,)
        which_f              = flat_idx % 4                       # (B,)

        x_new, y_new         = pos[:, 0].clone(), pos[:, 1].clone()
        mask0                = (which_f == 0)                     # left
        mask1                = (which_f == 1)                     # right
        mask2                = (which_f == 2)                     # bottom
        mask3                = (which_f == 3)                     # top
        if mask0.any():
            x_new[mask0]     = (xmin[which_r[mask0]] - ar)
        if mask1.any():
            x_new[mask1]     = (xmax[which_r[mask1]] + ar)
        if mask2.any():
            y_new[mask2]     = (ymin[which_r[mask2]] - ar)
        if mask3.any():
            y_new[mask3]     = (ymax[which_r[mask3]] + ar)

        return torch.stack([x_new, y_new], dim=-1)

    def step(self, action, inputs, endpoint_load=None):
        """
        action:        torch.Tensor (B,2)
        inputs:        torch.Tensor (B,C)  -> canais da task (tx,ty,go,...)
        endpoint_load: torch.Tensor | None (B,2)  -> ignorado aqui (não usamos cargas)
        """
        # clamp de velocidade no device atual
        a                  = self._clamp_norm_torch(action, self.speed_limit)          # (B,2)
        new_pos            = self._pos_t + self.dt * a                                 # (B,2)
        new_pos            = self._resolve_collision_torch(new_pos)                    # colisão AABB

        self._pos_t        = new_pos
        obs                = self._pos_t
        joint_state        = torch.zeros((obs.shape[0], 0), dtype=obs.dtype, device=obs.device)
        info               = {"states": {"xy": obs, "joint": joint_state}}
        reward             = None
        terminated         = False
        truncated          = False
        return obs, reward, terminated, truncated, info

    # ---------- geração de dados offline (permanece em NumPy) ----------
    def generate_dataset(self, n_samples: int):
        rng                 = np.random.default_rng()
        T                   = self.n_timesteps
        t                   = self.t_ms  # vetor de tempo (ms) do próprio ambiente, len == T

        # Canais: [0]=tx_norm, [1]=ty_norm, [2]=go, [3]=phase_init, [4]=phase_target,
        #         [5]=phase_delay, [6]=phase_move, [7]=maze_id (constante no trial)
        C                   = 8
        inputs              = np.zeros((n_samples, T, C), dtype=np.float32)
        targets             = np.zeros((n_samples, T, 2), dtype=np.float32)
        ics                 = np.zeros((n_samples, 2), dtype=np.float32)
        conds               = np.zeros((n_samples, 2), dtype=np.int32)  # (maze_id, ver)

        target_on_all, delay_end_all, go_all, max_abs_all = [], [], [], []

        for i in range(n_samples):
            # --- escolhe um maze ---
            key, _maze_dict   = self._sample_key(rng)
            X                 = self.traj_npz[f"{key}_X"].astype(np.float32)
            Y                 = self.traj_npz[f"{key}_Y"].astype(np.float32)
            t_src             = self.traj_npz[f"{key}_t_ms"].astype(np.float32)

            # --- alinha para a grade t ---
            Xr, Yr            = self._align_traj(X, Y, t_src=t_src, t_dst=t)

            # normalização global para os canais de entrada (apenas visual/escala)
            max_abs           = max(1e-6, float(np.nanmax(np.abs(np.stack([Xr, Yr], axis=-1)))))
            Xn                = Xr / max_abs
            Yn                = Yr / max_abs

            # posições de interesse
            p0                = np.array([Xr[0],   Yr[0]], dtype=np.float32)  # start
            p0_n              = np.array([Xn[0],   Yn[0]], dtype=np.float32)  # start (norm.)
            goal              = np.array([Xr[-1],  Yr[-1]], dtype=np.float32) # alvo final
            goal_n            = np.array([Xn[-1],  Yn[-1]], dtype=np.float32)

            # --- agenda das fases (índices, sem overlap) ---
            target_on_idx     = int(rng.integers(low=max(5, T//10), high=min(T//4, T-200)))
            delay_len         = int(rng.integers(low=30, high=90))
            delay_end_idx     = min(T-2, target_on_idx + delay_len)
            go_idx            = delay_end_idx
            move_idx          = go_idx

            # --- fases one-hot ---
            phase_init        = np.zeros(T, np.float32); phase_init[:target_on_idx]                = 1.0
            phase_target      = np.zeros(T, np.float32); phase_target[target_on_idx:delay_end_idx] = 1.0
            phase_delay       = np.zeros(T, np.float32); phase_delay[delay_end_idx:go_idx]         = 1.0
            phase_move        = np.zeros(T, np.float32); phase_move[move_idx:]                     = 1.0

            # --- canal go ---
            go                = np.zeros(T, np.float32); go[move_idx:] = 1.0

            # --- entradas tx,ty por fase ---
            tx = np.zeros(T, np.float32); ty = np.zeros(T, np.float32)
            tx[phase_init   > 0.5] = p0_n[0];   ty[phase_init   > 0.5] = p0_n[1]
            tx[phase_target > 0.5] = goal_n[0]; ty[phase_target > 0.5] = goal_n[1]
            tx[phase_delay  > 0.5] = goal_n[0]; ty[phase_delay  > 0.5] = goal_n[1]
            tx[phase_move   > 0.5] = goal_n[0]; ty[phase_move   > 0.5] = goal_n[1]

            # --- TARGETS (o que o modelo deve produzir) ---
            tgt_xy            = np.tile(p0[None, :], (T, 1))  # tudo em p0
            L                 = T - move_idx
            if L > 0:
                segX         = Xr[:L]; segY = Yr[:L]
                tgt_xy[move_idx:, 0] = segX
                tgt_xy[move_idx:, 1] = segY

            # --- escreve tensores nos buffers ---
            inputs[i, :, 0]   = tx
            inputs[i, :, 1]   = ty
            inputs[i, :, 2]   = go
            inputs[i, :, 3]   = phase_init
            inputs[i, :, 4]   = phase_target
            inputs[i, :, 5]   = phase_delay
            inputs[i, :, 6]   = phase_move

            mid               = int(key.split("_")[0].replace("maze", ""))
            ver               = int(key.split("_")[1].replace("ver", ""))
            inputs[i, :, 7]   = float(mid)  # opcional (id bruto)

            targets[i]        = tgt_xy
            ics[i]            = p0
            conds[i]          = np.array([mid, ver], dtype=np.int32)

            # guarda para "extra"
            target_on_all.append(target_on_idx)
            delay_end_all.append(delay_end_idx)
            go_all.append(go_idx)
            max_abs_all.append(max_abs)

        # "extra" estritamente numérico (HDF5-friendly)
        extra                = np.column_stack([
            np.asarray(target_on_all, dtype=np.int32),
            np.asarray(delay_end_all, dtype=np.int32),
            np.asarray(go_all,        dtype=np.int32),
            np.asarray(max_abs_all,   dtype=np.float32),
        ]).astype(np.float32)  # (N,4)

        dataset_dict         = dict(
            ics            = ics,
            inputs         = inputs,                           # [tx_n,ty_n,go,init,target,delay,move,maze_id]
            inputs_to_env  = np.zeros((n_samples, T, 0), dtype=np.float32),  # paredes ficam no ambiente
            targets        = targets,                          # [x,y] — hold até go, depois trajetória
            true_inputs    = inputs.copy(),
            conds          = conds,                            # (maze_id, version)
            extra          = extra,                            # (N,4) float32
        )
        extra_dict           = {}
        return dataset_dict, extra_dict
