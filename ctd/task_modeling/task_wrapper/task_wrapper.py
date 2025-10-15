import pytorch_lightning as pl
import torch
from gymnasium import Env
from torch import nn


class TaskTrainedWrapper(pl.LightningModule):
    """Wrapper for a task trained model

    Handles the training and validation steps for a task trained model.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay:  float,
        input_size=None,
        output_size=None,
        task_env: Env = None,
        model: nn.Module = None,
    ):
        super().__init__()
        self.task_env      = task_env
        self.model         = model
        self.input_size    = input_size
        self.output_size   = output_size
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay

        self.save_hyperparameters()

    def set_environment(self, task_env: Env):
        """Set the environment for the training pipeline"""
        self.task_env    = task_env
        self.input_size  = task_env.context_inputs.shape[0] + task_env.observation_space.shape[0]
        self.output_size = task_env.action_space.shape[0]
        self.loss_func   = task_env.loss_func
        self.state_label = getattr(task_env, "state_label", "xy")
        self.dynamic_noise = getattr(task_env, "dynamic_noise", 0.0)

    def set_model(self, model: nn.Module):
        """Set the model for the training pipeline"""
        self.model       = model
        self.latent_size = model.latent_size

    def configure_optimizers(self):
        """Configure the optimizer"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward_step_coupled(self, env_states, context_inputs, rnn_hidden, joint_state):
        """Forward step for coupled environment combining the RNN and environment"""
        inputs             = torch.hstack((env_states, context_inputs))
        action, rnn_hidden = self.model(inputs, rnn_hidden)
        env_states, _, terminated, _, info = self.task_env.step(
            action=action, inputs=context_inputs
        )
        joint_state        = info["states"]["joint"]
        return action, rnn_hidden, env_states, joint_state

    @torch.no_grad()
    def _ensure_device_like(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        return t if (t.device == device) else t.to(device)

    def _extract_maze_from_conds(self, conds: torch.Tensor):
        """Retorna (maze_id, version) como tensores long; se houver múltiplos no batch, usa o primeiro e avisa."""
        if conds is None:
            return None, None
        # conds shape: (B,2) -> [maze_id, version]
        maze_ids = conds[:, 0].to(torch.long)
        vers     = conds[:, 1].to(torch.long)

        # checagem de homogeneidade do batch
        # (ideal: usar um sampler que agrupe por maze_id/version)
        if maze_ids.unique().numel() > 1 or vers.unique().numel() > 1:
            # loga apenas uma vez para não poluir
            if not hasattr(self, "_warned_mixed_maze"):
                self._warned_mixed_maze = True
                self.print("[TaskTrainedWrapper] Aviso: batch possui múltiplos (maze_id, version). "
                           "Usando o primeiro par para o reset do env. Considere agrupar batches por maze.")
        return maze_ids, vers

    def forward(self, ics, inputs, inputs_to_env=None, conds=None):
        """Pass data through the model

        args:
            ics (torch.Tensor):     initial conditions (B, ...)
            inputs (torch.Tensor):  model inputs         (B, T, C)
            inputs_to_env:          extra env inputs     (B, T, ?)
            conds (torch.Tensor):   (B,2) -> [maze_id, version]
        """
        device = self.device
        ics    = ics.to(device)
        inputs = inputs.to(device)
        if inputs_to_env is not None:
            inputs_to_env = inputs_to_env.to(device)

        batch_size = ics.shape[0]

        # Maze (id, ver) vindos do dataset
        maze_id, version = self._extract_maze_from_conds(conds)

        # If a coupled environment, set the environment state
        if self.task_env.coupled_env:
            options = {"ic_state": ics, "device": device}
            if maze_id is not None and version is not None:
                options["maze_id"] = maze_id
                options["version"] = version
            env_states, info = self.task_env.reset(batch_size=batch_size, options=options)
            if isinstance(env_states, torch.Tensor):
                env_states = self._ensure_device_like(env_states, device)
            env_state_list = []
            joints         = []
        else:
            env_states     = None
            env_state_list = None

        # Hidden
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(batch_size=batch_size).to(device)
        else:
            hidden = torch.zeros(batch_size, self.latent_size, device=device)

        latents, controlled, actions = [], [], []

        count      = 0
        terminated = False
        while not terminated and len(controlled) < self.task_env.n_timesteps:
            # Build inputs
            if self.task_env.coupled_env:
                env_states_t = env_states if env_states.device == device else env_states.to(device)
                step_inputs  = inputs[:, count, :]
                model_input  = torch.hstack((env_states_t, step_inputs))
            else:
                model_input  = inputs[:, count, :]

            # optional dynamic noise
            if self.dynamic_noise > 0:
                model_input = model_input + torch.randn_like(model_input) * self.dynamic_noise

            # RNN step
            action, hidden = self.model(model_input, hidden)

            # Environment step (coupled)
            if self.task_env.coupled_env:
                kwargs = {}
                if inputs_to_env is not None:
                    kwargs["endpoint_load"] = inputs_to_env[:, count, :]
                env_states, _, terminated, _, info = self.task_env.step(
                    action=action, inputs=inputs[:, count, :], **kwargs
                )

                if isinstance(env_states, torch.Tensor):
                    env_states = self._ensure_device_like(env_states, device)
                state_t = self._ensure_device_like(info["states"][self.state_label], device)
                joint_t = self._ensure_device_like(info["states"]["joint"],      device)

                controlled.append(state_t)
                joints.append(joint_t)
                actions.append(action)
                env_state_list.append(env_states)
            else:
                controlled.append(action)
                actions.append(action)

            latents.append(hidden)
            count += 1

        # Stack outputs
        controlled = torch.stack(controlled, dim=1)
        latents    = torch.stack(latents,    dim=1)
        actions    = torch.stack(actions,    dim=1)
        if self.task_env.coupled_env:
            states = torch.stack(env_state_list, dim=1)
            joints = torch.stack(joints,        dim=1)
        else:
            states = None
            joints = None

        return {
            "controlled": controlled,
            "latents":    latents,
            "actions":    actions,
            "states":     states,
            "joints":     joints,
        }

    def training_step(self, batch, batch_ix):
        ics           = batch[0]
        inputs        = batch[1]
        targets       = batch[2]
        conds         = batch[4]
        extras        = batch[5]
        inputs_to_env = batch[6]

        out = self.forward(ics, inputs, inputs_to_env, conds=conds)

        # garanta que tudo está no mesmo device para a loss
        device = self.device
        loss_dict = {
            "controlled": out["controlled"],
            "latents":    out["latents"],
            "actions":    out["actions"],
            "targets":    targets.to(device),
            "inputs":     inputs.to(device),
            "conds":      conds.to(device),
            "extra":      extras,                   # se for NumPy, a loss deve lidar; caso contrário converta aqui
            "epoch":      self.current_epoch,
        }
        loss_all = self.loss_func(loss_dict)
        self.log("train/loss", loss_all, prog_bar=True, on_step=True, on_epoch=True)
        return loss_all

    def validation_step(self, batch, batch_ix):
        ics           = batch[0]
        inputs        = batch[1]
        targets       = batch[2]
        conds         = batch[4]
        extras        = batch[5]
        inputs_to_env = batch[6]

        out = self.forward(ics, inputs, inputs_to_env=inputs_to_env, conds=conds)

        device = self.device
        loss_dict = {
            "controlled": out["controlled"],
            "actions":    out["actions"],
            "latents":    out["latents"],
            "targets":    targets.to(device),
            "inputs":     inputs.to(device),
            "conds":      conds.to(device),
            "extra":      extras,
            "epoch":      self.current_epoch,
        }
        loss_all = self.loss_func(loss_dict)
        self.log("valid/loss", loss_all, prog_bar=True, on_step=False, on_epoch=True)
        return loss_all
