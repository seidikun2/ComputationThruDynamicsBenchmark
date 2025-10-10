# Class to generate training data for task-trained RNN that does 3 bit memory task
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from motornet.environment import Environment
from numpy import ndarray
from torch._tensor import Tensor

from ctd.task_modeling.task_env.loss_func import NBFFLoss, RandomTargetLoss

import json
from pathlib import Path


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
        self.n_timesteps = n_timesteps
        self.noise = noise

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
        ics: initial conditions
        inputs: inputs to the environment
        targets: targets of the environment
        conds: conditions information (if applicable)
        extra: extra information
        #----------Optional keys----------
        true_inputs: true inputs to the environment (if applicable)
        true_targets: true targets of the environment (if applicable)
        phase_dict: phase information (if applicable)
        """

        pass


class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy environment where the goal is to flip the required bit.
    """

    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        n=1,
        switch_prob=0.01,
        transition_blind=4,
        dynamic_noise=0,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = f"{n}BFF"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(n,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )
        self.n = n
        self.state = np.zeros(n)
        self.input_labels = [f"Input {i}" for i in range(n)]
        self.output_labels = [f"Output {i}" for i in range(n)]
        self.noise = noise
        self.dynamic_noise = dynamic_noise
        self.coupled_env = False
        self.switch_prob = switch_prob
        self.transition_blind = transition_blind
        self.loss_func = NBFFLoss(transition_blind=transition_blind)

    def step(self, action):
        # Generates state update given an input to the flip-flop
        for i in range(self.n):
            if action[i] == 1:
                self.state[i] = 1
            elif action[i] == -1:
                self.state[i] = 0

    def generate_trial(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand = np.random.random(size=(self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :] = 0

        # Generate the desired outputs given the inputs
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :] = self.state

        # Add noise to the inputs for the trial
        true_inputs = inputs
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def reset(self):
        self.state = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.n))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.zeros(shape=(n_samples, 1)),
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def render(self):
        inputs, states, _ = self.generate_trial()
        fig1, axes = plt.subplots(nrows=self.n + 1, ncols=1, sharex=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n))
        for i in range(self.n):
            axes[i].plot(states[:, i], color=colors[i])
            axes[i].set_ylabel(f"State {i}")
            axes[i].set_ylim(-0.2, 1.2)
        ax2 = axes[-1]
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
            fig = plt.figure(figsize=(5 * n_trials, 5))
            # Make colormap for the timestep in a trial
            for i in range(n_trials):

                ax = fig.add_subplot(1, n_trials, i + 1, projection="3d")
                inputs, states, _ = self.generate_trial()
                ax.plot(states[:, 0], states[:, 1], states[:, 2])
                ax.set_xlabel("Bit 1")
                ax.set_ylabel("Bit 2")
                ax.set_zlabel("Bit 3")
                ax.set_title(f"Trial {i+1}")
            plt.tight_layout()
            plt.show()


class RandomTarget(Environment):
    """A reach to a random target from a random starting position with a delay period.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass.
        This is the network that will perform the task.

        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative
        contribution to the default muscle L2 loss.

        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless

        self.dataset_name = "RandomTarget"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["TargetX", "TargetY", "GoCue"]
        self.output_labels = ["Pec", "Delt", "Brad", "TriLong", "Biceps", "TriLat"]
        self.context_inputs = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        self.coupled_env = True
        self.state_label = "fingertip"

        pos_weight = kwargs.get("pos_weight", 1.0)
        act_weight = kwargs.get("act_weight", 1.0)

        self.bump_mag_low = kwargs.get("bump_mag_low", 5)
        self.bump_mag_high = kwargs.get("bump_mag_high", 10)

        self.loss_func = RandomTargetLoss(
            position_loss=nn.MSELoss(), pos_weight=pos_weight, act_weight=act_weight
        )

    def generate_dataset(self, n_samples):
        # Make target circular, change loss function to be pinned at zero
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 3))

        goal_list = []
        go_cue_list = []
        target_on_list = []
        catch_trials = []
        ext_inputs_list = []

        for i in range(n_samples):
            catch_trial = np.random.choice([0, 1], p=[0.8, 0.2])
            bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])
            move_bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])

            target_on = np.random.randint(10, 30)
            go_cue = np.random.randint(target_on, self.n_timesteps)
            if move_bump_trial:
                bump_time = np.random.randint(go_cue, go_cue + 40)
            else:
                bump_time = np.random.randint(0, self.n_timesteps - 30)
            bump_duration = np.random.randint(15, 30)
            bump_theta = np.random.uniform(0, 2 * np.pi)
            bump_mag = np.random.uniform(self.bump_mag_low, self.bump_mag_high)

            target_on_list.append(target_on)

            info = self.generate_trial_info()
            initial_state.append(info["ics_joint"])
            initial_state_xy = info["ics_xy"]

            env_inputs_mat = np.zeros((self.n_timesteps, 2))
            if bump_trial:
                bump_end = min(bump_time + bump_duration, self.n_timesteps)
                env_inputs_mat[bump_time:bump_end, :] = np.array(
                    [bump_mag * np.cos(bump_theta), bump_mag * np.sin(bump_theta)]
                )

            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))
            if catch_trial:
                go_cue = -1
                goal_matrix[:, :] = initial_state_xy
            else:
                inputs[i, go_cue:, 2] = 1

                goal_matrix[:go_cue, :] = initial_state_xy
                goal_matrix[go_cue:, :] = torch.squeeze(info["goal"])

            go_cue_list.append(go_cue)
            inputs[i, target_on:, 0:2] = info["goal"]

            catch_trials.append(catch_trial)
            goal_list.append(goal_matrix)
            ext_inputs_list.append(env_inputs_mat)

        go_cue_list = np.array(go_cue_list)
        target_on_list = np.array(target_on_list)
        env_inputs = np.stack(ext_inputs_list, axis=0)
        extra = np.stack((target_on_list, go_cue_list), axis=1)
        conds = np.array(catch_trials)

        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "inputs_to_env": env_inputs,
            "targets": goal_list,
            "conds": conds,
            "extra": extra,
            "true_inputs": inputs,
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """
        sho_limit = [0, 135]  # mechanical constraints - used to be -90 180
        elb_limit = [0, 155]
        sho_ang = np.deg2rad(np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30))
        elb_ang = np.deg2rad(np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30))

        sho_ang_targ = np.deg2rad(
            np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30)
        )
        elb_ang_targ = np.deg2rad(
            np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30)
        )

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))
        ang_targ = torch.tensor(np.array([sho_ang_targ, elb_ang_targ, 0, 0]))

        target_pos = self.joint2cartesian(
            torch.tensor(ang_targ, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
        )
        return info

    def set_goal(
        self,
        goal: torch.Tensor,
    ):
        """
        Sets the goal of the task. This is the target position of the effector.
        """
        self.goal = goal

    def get_obs(self, action=None, deterministic: bool = False) -> Union[Tensor, ndarray]:
        self.update_obs_buffer(action=action)

        obs_as_list = [
            self.obs_buffer["vision"][0],
            self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][: self.action_frame_stacking]

        obs = torch.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)

    def reset(
        self,
        batch_size: int = 1,
        options: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> tuple[Any, dict[str, Any]]:

        """
        Uses the :meth:`Environment.reset()` method of the parent class
        :class:`Environment` that can be overwritten to change the returned data.
        Here the goals (`i.e.`, the targets) are drawn from a random uniform
        distribution across the full joint space.
        """
        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        # Make self.obs_noise a list
        self._set_generator(seed=seed)
        # if ic_state is in options, use that
        if options is not None and "deterministic" in options.keys():
            deterministic = options["deterministic"]
        else:
            deterministic = False
        if options is not None and "ic_state" in options.keys():
            ic_state_shape = np.shape(self.detach(options["ic_state"]))
            if ic_state_shape[0] > 1:
                batch_size = ic_state_shape[0]
            ic_state = options["ic_state"]
        else:
            ic_state = self.q_init

        if options is not None and "target_state" in options.keys():
            self.goal = options["target_state"]
        else:
            sho_ang = np.random.uniform(
                sho_limit[0] + 20, sho_limit[1] - 20, size=batch_size
            )
            elb_ang = np.random.uniform(
                elb_limit[0] + 20, elb_limit[1] - 20, size=batch_size
            )
            sho_vel = np.zeros(batch_size)
            elb_vel = np.zeros(batch_size)
            angs = np.stack((sho_ang, elb_ang, sho_vel, elb_vel), axis=1)
            self.goal = self.joint2cartesian(
                torch.tensor(angs, dtype=torch.float32, device=self.device)
            ).chunk(2, dim=-1)[0]

        options = {
            "batch_size": batch_size,
            "joint_state": ic_state,
        }
        self.effector.reset(options=options)

        self.elapsed = 0.0

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
        return obs, info

# ============================================================
# MazeWorld2D: ambiente acoplado com colisões retangulares
# - estado: XY
# - ação do modelo: velocidade XY (ou delta XY)
# - step aplica integração + colisão com barreiras AABB
# - lê mazes.json + traj_means.npz exportados
# Compatível com TaskTrainedWrapper (coupled_env=True)
# ============================================================
# ============================================================
# MazeWorld2D (corrigido): alinhamento por interpolação,
# maze/versão fixos por episódio e step determinístico.
# ============================================================

class MazeTask:
    """Ambiente 2D com paredes retangulares (AABB) e alvo XY."""

    def __init__(
        self,
        export_dir: Union[str, Path],
        n_timesteps: int,
        dt: float = 0.01,
        speed_limit: float = 100.0,
        agent_radius: float = 0.0,
        t_ms: Sequence[int] = np.arange(-50, 451),   # grade desejada, em ms
        use_maze_ids: Optional[Sequence[int]] = None,
        use_versions: Optional[Sequence[int]] = None,
        go_cue_at_zero: bool = True,
        add_xy_noise: float = 0.0,
    ):
        self.dataset_name  = "MazeWorld2D"
        self.n_timesteps   = int(n_timesteps)
        self.dt            = float(dt)
        self.speed_limit   = float(speed_limit)
        self.agent_radius  = float(agent_radius)
        self.coupled_env   = True
        self.state_label   = "xy"
        self.dynamic_noise = 0.0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.context_inputs    = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

        self.loss_func = nn.MSELoss()

        export_dir          = Path(export_dir)
        with open(export_dir / "mazes.json", "r", encoding="utf-8") as f:
            self.mazes = json.load(f)
        self.traj_npz       = np.load(export_dir / "traj_means.npz")

        self.use_maze_ids   = set(use_maze_ids) if use_maze_ids is not None else None
        self.use_versions   = set(use_versions) if use_versions is not None else None
        self.keys           = self._collect_valid_keys()
        any_k               = next(iter(self.keys))
        self.t_ms_export    = self.traj_npz[f"{any_k}_t_ms"].astype(int)
        self.t_ms           = np.asarray(t_ms, dtype=int)
        self.go_cue_at_zero = bool(go_cue_at_zero)
        self.add_xy_noise   = float(add_xy_noise)

        # estado interno
        self._pos        = None     # (B,2)
        self._goal       = None     # (B,2)
        self._rects      = None     # barreiras do maze corrente
        self._cur_key    = None     # "maze{mid}_ver{ver}"
        self._terminated = False

        # se n_timesteps não bate com len(t_ms), force T = len(t_ms)
        if self.n_timesteps != len(self.t_ms):
            self.n_timesteps = len(self.t_ms)

    # ---------- utils geométricos ----------

    @staticmethod
    def _clamp_norm(v, maxnorm):
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
        f = np.minimum(1.0, maxnorm / n)
        return v * f

    @staticmethod
    def _rects_from_json(maze_dict):
        rects = []
        for cx, cy, hw, hh in maze_dict.get("barriers", []):
            xmin, xmax = cx - hw, cx + hw
            ymin, ymax = cy - hh, cy + hh
            rects.append([xmin, xmax, ymin, ymax])
        return rects

    def _resolve_collision(self, p, rects):
        x, y = p
        for (xmin, xmax, ymin, ymax) in rects:
            if (xmin - self.agent_radius) <= x <= (xmax + self.agent_radius) and \
               (ymin - self.agent_radius) <= y <= (ymax + self.agent_radius):
                dx_left  = abs(x - (xmin - self.agent_radius))
                dx_right = abs((xmax + self.agent_radius) - x)
                dy_bot   = abs(y - (ymin - self.agent_radius))
                dy_top   = abs((ymax + self.agent_radius) - y)
                m = min(dx_left, dx_right, dy_bot, dy_top)
                if   m == dx_left:  x = xmin - self.agent_radius
                elif m == dx_right: x = xmax + self.agent_radius
                elif m == dy_bot:   y = ymin - self.agent_radius
                else:               y = ymax + self.agent_radius
        return np.array([x, y], dtype=np.float32)

    def _collect_valid_keys(self):
        keys = []
        for k in self.mazes.keys():  # formato "mazeId__version"
            mid_str, ver_str = k.split("__")
            mid, ver = int(mid_str), int(ver_str)
            if (self.use_maze_ids is not None) and (mid not in self.use_maze_ids):
                continue
            if (self.use_versions is not None) and (ver not in self.use_versions):
                continue
            base = f"maze{mid}_ver{ver}"
            if (f"{base}_X" in self.traj_npz) and (f"{base}_Y" in self.traj_npz) and (f"{base}_t_ms" in self.traj_npz):
                keys.append(base)
        if not keys:
            raise RuntimeError("Sem chaves válidas no NPZ para os filtros.")
        return keys

    def _sample_key(self, rng):
        key = rng.choice(self.keys)
        mid = int(key.split("_")[0].replace("maze", ""))
        ver = int(key.split("_")[1].replace("ver", ""))
        maze_json_key = f"{mid}__{ver}"
        return key, self.mazes[maze_json_key]

    # ---------- alinhamento de trajetórias ----------

    def _align_traj(self, X_src, Y_src, t_src, t_dst):
        """Retorna X,Y interpolados em t_dst (mesma length de t_dst)."""
        X_src = np.asarray(X_src, dtype=np.float32)
        Y_src = np.asarray(Y_src, dtype=np.float32)
        t_src = np.asarray(t_src, dtype=np.float32)
        t_dst = np.asarray(t_dst, dtype=np.float32)

        # interp 1D; para fora do suporte, segura borda
        x_al = np.interp(t_dst, t_src, X_src, left=X_src[0], right=X_src[-1]).astype(np.float32)
        y_al = np.interp(t_dst, t_src, Y_src, left=Y_src[0], right=Y_src[-1]).astype(np.float32)
        return x_al, y_al

    # ---------- API esperada pelo wrapper ----------

    def reset(self, batch_size: int = 1, options: Optional[dict] = None):
        rng = np.random.default_rng()
        self._terminated = False

        # escolhe UM maze/versão e guarda para o episódio
        key, maze_dict = self._sample_key(rng)
        self._cur_key  = key
        self._rects    = self._rects_from_json(maze_dict)

        X = self.traj_npz[f"{key}_X"].astype(np.float32)
        Y = self.traj_npz[f"{key}_Y"].astype(np.float32)
        t = self.traj_npz[f"{key}_t_ms"].astype(np.float32)

        Xr, Yr = self._align_traj(X, Y, t_src=t, t_dst=self.t_ms)

        p0 = np.array([Xr[0], Yr[0]], dtype=np.float32)
        p0 = self._resolve_collision(p0, self._rects)

        self._pos  = np.tile(p0[None, :], (batch_size, 1))
        self._goal = np.tile(np.array([Xr[-1], Yr[-1]], np.float32), (batch_size, 1))

        obs = torch.from_numpy(self._pos).float()
        info = {"states": {"xy": obs, "joint": None}}
        return obs, info

    def step(self, action, inputs):
        a = action.detach().cpu().numpy().astype(np.float32)
        a = self._clamp_norm(a, self.speed_limit)

        new_pos = self._pos + self.dt * a
        for b in range(new_pos.shape[0]):
            new_pos[b, :] = self._resolve_collision(new_pos[b, :], self._rects)

        self._pos = new_pos
        obs = torch.from_numpy(self._pos).float()
        terminated = False
        truncated = False
        info = {"states": {"xy": obs, "joint": None}}
        reward = None
        return obs, reward, terminated, truncated, info
    
    def generate_dataset(self, n_samples: int):
        rng = np.random.default_rng()
        T = self.n_timesteps
        t = self.t_ms  # vetor de tempo (ms) do próprio ambiente, len == T
    
        # Canais: [0]=tx_norm, [1]=ty_norm, [2]=go,
        #         [3]=phase_init, [4]=phase_target, [5]=phase_delay, [6]=phase_move,
        #         [7]=maze_id (constante no trial)
        C = 8
        inputs  = np.zeros((n_samples, T, C), dtype=np.float32)
        targets = np.zeros((n_samples, T, 2), dtype=np.float32)
        ics     = np.zeros((n_samples, 2), dtype=np.float32)
        conds   = np.zeros((n_samples, 2), dtype=np.int32)   # (maze_id, ver)
        extras  = []
    
        for i in range(n_samples):
            # --- escolhe um maze ---
            key, _maze_dict = self._sample_key(rng)
            X = self.traj_npz[f"{key}_X"].astype(np.float32)
            Y = self.traj_npz[f"{key}_Y"].astype(np.float32)
            t_src = self.traj_npz[f"{key}_t_ms"].astype(np.float32)
    
            # --- alinha para a grade t ---
            Xr, Yr = self._align_traj(X, Y, t_src=t_src, t_dst=t)
    
            # normalização global para os canais de entrada (apenas visual/escala)
            max_abs = max(1e-6, float(np.nanmax(np.abs(np.stack([Xr, Yr], axis=-1)))))
            Xn = Xr / max_abs
            Yn = Yr / max_abs
    
            # posições de interesse
            p0   = np.array([Xr[0],   Yr[0]], dtype=np.float32)     # start
            p0_n = np.array([Xn[0],   Yn[0]], dtype=np.float32)     # start (norm.)
            goal = np.array([Xr[-1],  Yr[-1]], dtype=np.float32)    # alvo final
            goal_n = np.array([Xn[-1], Yn[-1]], dtype=np.float32)
    
            # --- agenda das fases (índices, sem overlap) ---
            # target_on_idx  < delay_end_idx  < go_idx
            # deixe janelas razoáveis; ajuste os ranges conforme seu t
            target_on_idx = int(rng.integers(low=max(5, T//10), high=min(T//4, T-200)))
            delay_len     = int(rng.integers(low=30, high=90))
            delay_end_idx = min(T-2, target_on_idx + delay_len)
            go_idx        = delay_end_idx  # go logo após delay (poderia ter outra amostra se quiser)
            move_idx      = go_idx
    
            # --- fases one-hot ---
            phase_init   = np.zeros(T, np.float32); phase_init[:target_on_idx]     = 1.0
            phase_target = np.zeros(T, np.float32); phase_target[target_on_idx:delay_end_idx] = 1.0
            phase_delay  = np.zeros(T, np.float32); phase_delay[delay_end_idx:go_idx]         = 1.0
            phase_move   = np.zeros(T, np.float32); phase_move[move_idx:]                    = 1.0
    
            # --- canal go (só na fase move) ---
            go = np.zeros(T, np.float32); go[move_idx:] = 1.0
    
            # --- entradas tx,ty por fase ---
            # INIT   : variáveis da task disponíveis (cursor/start) -> p0
            # TARGET : alvo visível (goal constante)
            # DELAY  : continua visível (goal constante)
            # MOVE   : pode manter goal constante (a rede já deve “seguir” o target via loss)
            tx = np.zeros(T, np.float32); ty = np.zeros(T, np.float32)
            tx[phase_init > 0.5]   = p0_n[0];   ty[phase_init > 0.5]   = p0_n[1]
            tx[phase_target > 0.5] = goal_n[0]; ty[phase_target > 0.5] = goal_n[1]
            tx[phase_delay > 0.5]  = goal_n[0]; ty[phase_delay > 0.5]  = goal_n[1]
            tx[phase_move > 0.5]   = goal_n[0]; ty[phase_move > 0.5]   = goal_n[1]
    
            # --- TARGETS (o que o modelo deve produzir) ---
            # Sem movimento até o go: manter p0 até go_idx,
            # depois “tocar” a trajetória do labirinto (Xr, Yr) a partir do início.
            tgt_xy = np.tile(p0[None, :], (T, 1))  # tudo em p0
            L = T - move_idx
            if L > 0:
                # encaixa a trajetória desde o começo em move_idx...
                segX = Xr[:L]; segY = Yr[:L]
                tgt_xy[move_idx:, 0] = segX
                tgt_xy[move_idx:, 1] = segY
    
            # --- escreve tensores nos buffers ---
            inputs[i, :, 0] = tx
            inputs[i, :, 1] = ty
            inputs[i, :, 2] = go
            inputs[i, :, 3] = phase_init
            inputs[i, :, 4] = phase_target
            inputs[i, :, 5] = phase_delay
            inputs[i, :, 6] = phase_move
    
            mid = int(key.split("_")[0].replace("maze", ""))
            ver = int(key.split("_")[1].replace("ver", ""))
            inputs[i, :, 7] = float(mid)  # opcional (id bruto)
    
            targets[i] = tgt_xy
            ics[i]     = p0
            conds[i]   = np.array([mid, ver], dtype=np.int32)
            extras.append(dict(key=key,
                               target_on_idx=int(target_on_idx),
                               delay_end_idx=int(delay_end_idx),
                               go_idx=int(go_idx),
                               max_abs=float(max_abs)))
    
        dataset_dict = dict(
            ics           = ics,
            inputs        = inputs,                           # [tx_n,ty_n,go,init,target,delay,move,maze_id]
            inputs_to_env = np.zeros((n_samples, T, 0), dtype=np.float32),  # paredes ficam no ambiente
            targets       = targets,                          # [x,y] — sem movimento até go, depois trajetória
            true_inputs   = inputs.copy(),
            conds         = conds,                            # (maze_id, version)
            extra         = np.array(extras, dtype=object).reshape(-1, 1),
        )
        extra_dict = {}
        return dataset_dict, extra_dict
