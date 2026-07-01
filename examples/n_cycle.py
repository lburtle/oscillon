import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from oscillon.topology import (
    BlockSoftSpec, init_block_params_from_adjacency,
    make_block_param_to_model, extract_block_adjacency, block_gate_matrix,
)
from oscillon.graph import MotifSpec, NetworkSpec, plot_network_graph
from oscillon.model import train
from oscillon.dynamics import simulate
from oscillon.readout import apply_readout

# ----------------------------------------------------------------------
N = 6                      # <-- number of nodes; change this to scale
dt, n_steps, burn_in = 0.1, 2000, 200
theta_init = 0.5           # livelier warm start than the 3-cycle default (see notes)
# ----------------------------------------------------------------------

script_dir = Path(__file__).parent
output_dir = script_dir / "images"
output_dir.mkdir(exist_ok=True)
key = jax.random.PRNGKey(0)


def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt


def make_targets(n, t, period, amplitudes=None, phases=None, offset=0.0):
    """n sinusoids sharing one period; per-channel amplitude + phase.
    Shared period is required for single-motif linear decodability."""
    omega = 2.0 * np.pi / period
    if phases is None:
        phases = [2.0 * np.pi * k / n for k in range(n)]          # even n-phase spread
    if amplitudes is None:
        amplitudes = [1.0 + 0.5 * (k % 3) for k in range(n)]      # some amplitude variety
    cols = [a * jnp.sin(omega * t + ph) + offset
            for a, ph in zip(amplitudes, phases, strict=True)]
    return jnp.stack(cols, axis=1)


# --- spec: single motif of size N ---
spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=0.10, delta=0.50, theta_init=theta_init,
)

# --- warm start from the discrete N-cycle ---
cycle = NetworkSpec([MotifSpec.cyclic(N)])
A_seed = cycle.to_torch_adjacency().numpy()
key, sub = jax.random.split(key)
params0 = init_block_params_from_adjacency(spec, sub, A_seed)
x0 = jnp.full((N,), 0.1)

# --- measure the warm-start cycle period, build target to match ---
t = jnp.arange(n_steps) * dt
p2m = make_block_param_to_model(spec)
W0, th0 = p2m(params0)
xs0 = simulate(W0, th0, x0, dt=dt, n_steps=n_steps)[burn_in:]
period = measure_period(xs0, dt)
if period is None:
    period = 2.0 * np.pi          # fallback if warm start didn't oscillate
    print("WARN: warm start not oscillating; using fallback period", period)
print(f"N={N}  matched target period = {period:.3f} time units "
      f"({period / dt:.1f} steps)")

targets = make_targets(N, t, period)

target_periods_in_window = 8          # how many full laps you want to see/fit
usable = target_periods_in_window * (period / dt)
n_steps = int(burn_in + usable)
print(f"N={N}  period={period/dt:.0f} steps  ->  n_steps={n_steps}")

# probe: rough period estimate with a generous fixed rollout
probe_steps = 4000
xs_probe = simulate(W0, th0, x0, dt=dt, n_steps=probe_steps)[burn_in:]
period = measure_period(xs_probe, dt)
if period is None:
    period = 2.0 * np.pi
n_steps = int(burn_in + 8 * (period / dt))   # real rollout sized to the cycle

# REBUILD t and targets at the new n_steps
t = jnp.arange(n_steps) * dt
targets = make_targets(N, t, period)

# --- train ---
result = train(
    spec, params0, x0, targets, key,
    dt=dt, n_steps=n_steps, burn_in=burn_in,
    warm=dict(n_iters=900, pop=128, sigma=0.1, lr=0.05),
    cryst=dict(n_iters=500, pop=128, sigma=0.05, lr=0.03),
)
print("final reward:", result.history[-1])

# --- evaluate on the trained network ---
xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)
y = apply_readout(xs, result.R, result.b)
tgt = targets[burn_in:]
mse = float(jnp.mean((y[burn_in:] - tgt) ** 2))
const = float(jnp.mean((tgt.mean(0) - tgt) ** 2))
print(f"readout MSE {mse:.4f}  vs constant-mean {const:.4f}")

# --- plots: vertical-offset stacking stays legible as N grows ---
cmap = plt.cm.viridis(np.linspace(0, 1, N))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 1.4 * N + 2), sharex=True)

spacing_a = 1.2 * float(jnp.max(xs))
for i in range(N):
    ax1.plot(xs[:, i] + i * spacing_a, color=cmap[i], lw=1.0)
ax1.axvline(burn_in, ls="--", c="gray", lw=0.8)
ax1.set_yticks([i * spacing_a for i in range(N)])
ax1.set_yticklabels([f"n{i}" for i in range(N)])
ax1.set_title(f"trained gCTLN activations (N={N})")

spacing_t = 1.2 * float(jnp.max(jnp.abs(targets)))
for i in range(N):
    ax2.plot(y[burn_in:, i] + i * spacing_t, color=cmap[i], lw=1.0)
    ax2.plot(tgt[:, i] + i * spacing_t, ls=":", c="k", lw=0.7)
ax2.set_yticks([i * spacing_t for i in range(N)])
ax2.set_yticklabels([f"ch{i}" for i in range(N)])
ax2.set_title("readout (color) vs target (dotted)")

plt.tight_layout()
plt.savefig(output_dir / f"scale_N{N}.png", dpi=130)
print(f"saved scale_N{N}.png")

# --- learned topology ---
A_learned = extract_block_adjacency(spec, result.params)
print("kept seed cycle:", np.array_equal(A_learned, A_seed))
G = block_gate_matrix(spec, result.params)
plot_network_graph(G, theta=np.asarray(result.theta),
                   save_path=str(output_dir / f"learned_graph_N{N}.png"))

from phase_portrait import plot_phase_portrait

if N == 3:
    # 3-cycle, view neurons 0 and 1 directly, several initial conditions
    x0s = [np.array([0.1, 0.0, 0.0]), np.array([0.5, 0.5, 0.0]), np.array([1.0, 0.2, 0.6])]
    plot_phase_portrait(result.W, result.theta, dims=(0, 1), x0s=x0s,
                        title="3-cycle phase portrait", save_path="images/phase_portraits/3-cycle_phase.png")

else:
    # any N: project onto the PCA plane of the learned limit cycle
    xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
    plot_phase_portrait(result.W, result.theta, dims=("pca", np.asarray(xs)), x0s=[x0],
                        title=f"N={N} limit cycle (PCA projection)", save_path=f"images/phase_portraits/{N}-cycle_phase.png")
