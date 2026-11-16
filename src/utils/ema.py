"""Exponential moving average of the trainable parameters."""

import torch


class EMA:
    """shadow = decay * shadow + (1 - decay) * param, over the parameters trainable at construction."""

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.params = dict(model.named_parameters())
        self.shadow = {n: p.detach().clone() for n, p in self.params.items() if p.requires_grad}
        self.backup = None

    @torch.no_grad()
    def update(self):
        for name, s in self.shadow.items():
            s.mul_(self.decay).add_(self.params[name].detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        """Put the averaged weights in the model, keeping the live ones for swap_out."""
        if self.backup is not None:
            raise RuntimeError("EMA.swap_in called twice without swap_out")
        self.backup = {n: self.params[n].detach().clone() for n in self.shadow}
        for name, s in self.shadow.items():
            self.params[name].copy_(s)

    @torch.no_grad()
    def swap_out(self):
        """Restore the live weights stored by swap_in."""
        if self.backup is None:
            raise RuntimeError("EMA.swap_out called without a preceding swap_in")
        for name, b in self.backup.items():
            self.params[name].copy_(b)
        self.backup = None

    def state_dict(self):
        return {n: s.detach().cpu().clone() for n, s in self.shadow.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.shadow):
            diff = sorted(set(state) ^ set(self.shadow))
            raise KeyError(f"EMA state keys differ from the tracked parameters: {diff[:5]}")
        for name, s in self.shadow.items():
            s.copy_(state[name].to(s.device))
