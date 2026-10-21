"""oscillon — command-line entry point.

Optional convenience CLI around the public API. Currently exposes a single
`simulate` subcommand that builds a random soft-topology CTLN and rolls it
out, as a smoke test that the installed package works end to end.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import jax
import numpy as np

from oscillon import SoftNetworkSpec, __version__, init_params, make_param_to_model, simulate


def _simulate(args: argparse.Namespace) -> None:
    key = jax.random.key(args.seed)
    key_init, key_x0 = jax.random.split(key)

    spec = SoftNetworkSpec(n=args.n, eps=args.eps, delta=args.delta)
    params = init_params(spec, key_init)
    param_to_model = make_param_to_model(spec)
    W, theta = param_to_model(params)

    x0 = jax.random.uniform(key_x0, (args.n,))
    xs = simulate(W, theta, x0, dt=args.dt, n_steps=args.steps)

    final = np.asarray(xs[-1])
    print(f"oscillon {__version__}: simulated n={args.n} for {args.steps} steps")
    print(f"final state: {np.round(final, 4)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oscillon", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("simulate", help="roll out a random CTLN and print its final state")
    sim.add_argument("-n", type=int, default=5, help="number of neurons (default: 5)")
    sim.add_argument("--eps", type=float, default=0.10, help="epsilon parameter (default: 0.10)")
    sim.add_argument("--delta", type=float, default=0.50, help="delta parameter (default: 0.50)")
    sim.add_argument("--dt", type=float, default=0.1, help="Euler step size (default: 0.1)")
    sim.add_argument("--steps", type=int, default=2000, help="number of integration steps (default: 2000)")
    sim.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    sim.set_defaults(func=_simulate)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
