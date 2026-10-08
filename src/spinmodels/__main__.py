"""Command-line temperature scans.

Example::

    python -m spinmodels ising --dim 2 -L 8 16 32 --tmin 2.0 --tmax 2.6 --nt 16 \\
        --algorithm wolff --out output/ising2d
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .models import MODELS
from .simulation import temperature_scan


def main(argv=None):
    p = argparse.ArgumentParser(prog="spinmodels", description=__doc__.split("\n")[0])
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--dim", type=int, default=None, help="lattice dimension (model default)")
    p.add_argument("-L", type=int, nargs="+", default=[16], help="linear sizes")
    p.add_argument("-q", type=int, default=3, help="Potts states")
    p.add_argument("-J", type=float, default=1.0)
    p.add_argument("--field", type=float, default=0.0, help="external field h")
    p.add_argument("--tmin", type=float, required=True)
    p.add_argument("--tmax", type=float, required=True)
    p.add_argument("--nt", type=int, default=16, help="number of temperatures")
    p.add_argument("--equil", type=int, default=1000, help="equilibration sweeps per T")
    p.add_argument("--measure", type=int, default=5000, help="measurements per T")
    p.add_argument("--every", type=int, default=1, help="sweeps between measurements")
    p.add_argument("--algorithm", default="metropolis")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output prefix for .npz and .png files")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    cls = MODELS[args.model]
    kwargs = {"J": args.J, "h": args.field, "seed": args.seed}
    if args.dim is not None:
        kwargs["dim"] = args.dim
    if args.model == "potts":
        kwargs["q"] = args.q

    temps = np.linspace(args.tmin, args.tmax, args.nt)
    results = []
    for L in args.L:
        model = cls(L, **kwargs)
        res = temperature_scan(
            model, temps, n_equil=args.equil, n_measure=args.measure,
            measure_every=args.every, algorithm=args.algorithm,
        )
        results.append(res)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            res.save(f"{args.out}_L{L}.npz")

    if not args.no_plot:
        import matplotlib

        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plotting import plot_scan

        Tc = cls.critical_temperature(results[0].dim, **results[0].params)
        fig, _ = plot_scan(results, Tc=Tc)
        if args.out:
            fig.savefig(f"{args.out}.png", dpi=150)
            print(f"Saved {args.out}.png")
        else:
            plt.show()


if __name__ == "__main__":
    main()
