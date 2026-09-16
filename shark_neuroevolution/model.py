"""Small MLP brain. Transparency over performance: plain Linear+ReLU."""
import numpy as np
import torch
import torch.nn as nn

import config


class Brain(nn.Module):
    def __init__(self, in_n=config.INPUT_NODES, hid=config.HIDDEN_NODES, out_n=config.OUTPUT_NODES):
        super().__init__()
        self.fc1 = nn.Linear(in_n, hid)
        self.fc2 = nn.Linear(hid, out_n)

    def forward(self, x):
        return torch.tanh(self.fc2(torch.relu(self.fc1(x))))

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():  # asarray avoids a copy when obs is already float32
            out = self(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
        return out.numpy()

    def get_weights(self):  # -> [W1, b1, W2, b2] as numpy, for visualizer
        return [p.detach().cpu().numpy() for p in
                (self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)]

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

    def save(self, path): torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, **kw):
        m = cls(**kw)
        # map_location: checkpoint saved on GPU must still load on CPU-only machines
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return m


if __name__ == "__main__":  # ponytail: one runnable check, no test framework
    b = Brain()
    assert b.act(np.zeros(8)).shape == (2,)
    c = Brain.crossover(b, Brain()); c.mutate()
    assert c.act(np.zeros(8)).shape == (2,)
    b.save("/tmp/_brain_test.pt"); Brain.load("/tmp/_brain_test.pt")
    print("model ok")
