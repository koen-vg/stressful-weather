# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Functionality to solve the operations of a given network.

For this given network, all components are set to be non-extendable
and load shedding is added to make the optimisation problem feasible.
The costs for load shedding are set high enough to avoid its usage as
much as possible.
"""

from pathlib import Path

import pypsa
from _helpers import configure_logging
from pypsa.descriptors import nominal_attrs, get_extendable_i
from pypsa.linopf import network_lopf


def set_nom_to_opt(n: pypsa.Network) -> None:
    """Overwrite nominal extendable capacities of `n` by optimal capacities.

    Modifies the argument `n`.

    """
    for c, attr in nominal_attrs.items():
        i = get_extendable_i(n, c)
        # Note: we make sure not to set any nominal capacities to NaN
        # by using `dropna()`.
        n.df(c).loc[i, attr] = n.df(c).loc[i, attr + "_opt"].dropna()


def set_extendable_false(n: pypsa.Network) -> None:
    """Set all technologies in `n` to non-extendable.

    Modifies the argument `n`.

    """
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + "_extendable"] = False


def add_load_shedding(n: pypsa.Network) -> None:
    """Add load shedding generator at every AC bus of `n`.

    Modifies the argument `n`.

    """
    n.add("Carrier", "load-shedding")
    buses_i = n.buses.query("carrier == 'AC'").index
    n.madd(
        "Generator",
        buses_i,
        " load shedding",
        bus=buses_i,
        carrier="load-shedding",
        # Marginal cost same as in highRES (Price and Zeyringer, 2022)
        marginal_cost=7.3e3,  # EUR/MWh
        p_nom=1e6,
    )


def set_weather(n_op: pypsa.Network, n_weather: pypsa.Network) -> None:
    """Set weather-dependent time series of `n_op` to those of `n_weather`.

    Modifies the argument `n_op`.

    """
    for c, attr in [
        ("Generator", "p_max_pu"),
        ("StorageUnit", "p_max_pu"),
        ("StorageUnit", "inflow"),
        ("Load", "p_set"),
    ]:
        n_op.pnl(c)[attr].loc[:, :] = n_weather.pnl(c)[attr].values


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log
    # file.
    if "snakemake" not in globals():
        configure_logging(snakemake)

    # Load the operation and weather networks
    n_op = pypsa.Network(snakemake.input.network_op)
    n_weather = pypsa.Network(snakemake.input.network_weather)

    # Set nominal to optimal capacities.
    set_nom_to_opt(n_op)

    # Set all components to be non-extenable.
    set_extendable_false(n_op)

    # Add load shedding to the network.
    add_load_shedding(n_op)

    # Set weather-dependent time series of n_op to those of n_weather.
    set_weather(n_op, n_weather)

    # At this point, the network `n_op` should have:
    # - Defined capacities (`p_nom`, `s_nom`, etc.) for all included
    #   technologies,
    # - No extendable technologies,
    # - Load shedding installed at all nodes,
    # - Possibly a single global constraint on CO2 emissions.

    # Prepare solving options.
    solver_options = snakemake.config["pypsa-eur"]["solving"]["solver"]
    solver_name = solver_options.pop("name")
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Solve the network.
    status, condition = network_lopf(
        n_op,
        solver_name=solver_name,
        solver_options=solver_options,
        solver_dir=tmpdir,
    )

    if status != "ok":
        raise RuntimeError(f"Solving for operations failed: {condition}.")

    # Export the result.
    n_op.export_to_netcdf(
        snakemake.output.network,
        compression={"zlib": True, "complevel": 4, "least_significant_digit": 3},
    )
