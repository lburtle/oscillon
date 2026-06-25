import jax
import jax.numpy as jnp

from oscillon.readout import fit_readout

n_steps, burn_in = 2000, 200
T = n_steps - burn_in
targets = jnp.zeros((T, 3))   # shapes are all that matter here, values irrelevant

xs_fake = jnp.zeros((128, T, 3))   # (pop, T, n)
R, b = jax.vmap(fit_readout, in_axes=(0, None, None))(xs_fake, targets, 1e-4)

print("R:", R.shape)   # expect (128, 3, 3)  -> (pop, m, n)
print("b:", b.shape)   # expect (128, 3)     -> (pop, m)
print("traced OK")