"""Variable-input MLP brain. Grows new sensors on level-up, keeping old weights."""
import numpy as np
import torch
import torch.nn as nn

import config


class Brain(nn.Module):
    def __init__(self, input_size, hidden_size=config.HIDDEN_NODES,
                 output_size=config.OUTPUT_NODES):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        with torch.no_grad():  # start near-silent: thin gray lines, gold is EARNED
            for p in self.parameters():
                p.uniform_(-0.25, 0.25)

    def forward(self, x):
        return torch.tanh(self.fc2(torch.relu(self.fc1(x))))

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():  # asarray avoids a copy when obs is already float32
            # .forward() directly, not self(x): skips nn.Module's hook-dispatch
            # wrapper (_call_impl/_wrapped_call_impl), which profiling showed
            # costs as much as the actual matmul for a model this tiny. Safe
            # here since this Brain never registers forward/backward hooks.
            out = self.forward(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
        return out.numpy()

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """Whole school through one forward pass (32x cheaper than per-mouse)."""
        with torch.no_grad():
            out = self.forward(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
        return out.numpy().ravel()

    def get_weights(self):  # for visualizer: w1 [hid, in], b1, w2 [out, hid]
        return {"w1": self.fc1.weight.detach().cpu().numpy(),
                "b1": self.fc1.bias.detach().cpu().numpy(),
                "w2": self.fc2.weight.detach().cpu().numpy()}

    def mutate(self, rate=config.MUTATION_RATE, strength=config.MUTATION_STRENGTH):
        with torch.no_grad():
            for p in self.parameters():
                mask = (torch.rand_like(p) < rate).float()
                p.add_(mask * torch.randn_like(p) * strength)

    @classmethod
    def crossover(cls, a: "Brain", b: "Brain") -> "Brain":
        child = cls(a.fc1.in_features, a.fc1.out_features, a.fc2.out_features)
        with torch.no_grad():
            for pc, pa, pb in zip(child.parameters(), a.parameters(), b.parameters()):
                mask = torch.rand_like(pa) < 0.5
                pc.copy_(torch.where(mask, pa, pb))
        return child

    def grow(self, new_size: int) -> "Brain":
        """New brain with extra input columns. Old weights preserved,
        fresh sensors start quiet (x0.1) so they don't scramble behavior."""
        child = Brain(new_size, self.fc1.out_features, self.fc2.out_features)
        with torch.no_grad():
            old = self.fc1.weight.data.shape[1]
            child.fc1.weight.data[:, :old] = self.fc1.weight.data
            child.fc1.weight.data[:, old:] *= 0.1
            child.fc1.bias.data = self.fc1.bias.data.clone()
            child.fc2.weight.data = self.fc2.weight.data.clone()
            child.fc2.bias.data = self.fc2.bias.data.clone()
        return child

    def save(self, path): torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, input_size, **kw):
        m = cls(input_size, **kw)
        # map_location: GPU-saved checkpoint must still load on CPU-only machines
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return m


if __name__ == "__main__":  # ponytail: one runnable check, no test framework
    b = Brain(1)
    assert b.act(np.ones(1)).shape == (1,)
    g = b.grow(4)  # level-up preserves old columns
    assert np.allclose(g.fc1.weight.data[:, :1].numpy(),
                       b.fc1.weight.data.numpy())
    c = Brain.crossover(g, Brain(4)); c.mutate()
    b.save("/tmp/_brain_test.pt"); Brain.load("/tmp/_brain_test.pt", 1)
    print("model ok")