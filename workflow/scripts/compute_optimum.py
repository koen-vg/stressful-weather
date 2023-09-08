# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optimising a PyPSA network with respect to its total system costs."""

import logging
import time
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging
from solve_network_linopy import prepare_network, solve_network
from workflow_utilities import parse_net_spec

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    n = pypsa.Network(snakemake.input.network)
    solving_options = snakemake.config["pypsa-eur"]["solving"]["options"]
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    opts = parse_net_spec(snakemake.wildcards.spec)["opts"].split("-")

    # Solve the network for the cost optimum and then get its
    # coordinates in the basis.
    logging.info("Compute initial, optimal solution.")
    t = time.time()
    n = prepare_network(n, solving_options)
    n, status = solve_network(
        n,
        snakemake.config["pypsa-eur"],
        solver_dir=tmpdir,
        opts=opts,
        solver_logfile=snakemake.log.solver,
    )
    logging.info(f"Optimisation took {time.time() - t:.2f} seconds.")

    # Check if the optimisation succeeded; if not we don't output
    # anything in order to make snakemake fail. Not checking for this
    # would result in an invalid (non-optimal) network being output.
    if status == "ok":
        # Write the result to the given output files. Save the objective
        # value for further processing.
        n.export_to_netcdf(snakemake.output.optimum)
        with open(snakemake.output.obj, "w") as f:
            f.write(str(n.objective))
