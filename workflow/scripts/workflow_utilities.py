# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utilities aiding the Snakemake workflow."""


import os
import re
from os.path import join

import yaml


def validate_configs(config_dir: str):
    """Check that every file in `config_dir` is well-formed.

    Specifically, every file must have a name of the form
    `config-<name>.yaml`, and must contain a top-level key `name:
    <name>`.
    """
    # Loop through the files in `config_dir`.
    for fn in os.listdir(config_dir):
        # Check that the name follows the required format.
        m = re.match(r"config-(.+).yaml", fn)
        if not m:
            raise ValueError(f"Found configuration file with bad name: {fn}")
        name = m.group(1)
        # Load the config.
        with open(join(config_dir, fn), "r") as f:
            contents = yaml.safe_load(f)
        # Check that the name given in the config matches the filename.
        if "name" not in contents:
            raise ValueError(f"Config file {fn} does not have a name key.")
        if contents["name"] != name:
            raise ValueError(f"Config file {fn} has bad name key: {contents['name']}")


def parse_net_spec(spec: str) -> dict:
    """Parse a network specification and return it as a dictionary."""
    # Define the individual regexes for all the different wildcards.
    rs = {
        "year": r"([0-9]+)(-[0-9]+)?(\+([0-9]+)(-[0-9]+)?)*",
        "simpl": r"[a-zA-Z0-9]*|all",
        "clusters": r"[0-9]+m?|all|[0-9]+-[0-9]+-[0-9]+",
        "ll": r"(v|c)([0-9\.]+|opt|all)|all",
        "opts": r"[-+a-zA-Z0-9\.]*",
    }
    # Make named groups out of the individual groups
    G = {n: f"(?P<{n}>{r})" for n, r in rs.items()}
    # Build the complete regex out of the individual groups
    full_regex_pypsa_eur = (
        f"({G['year']}_)?" f"{G['simpl']}_{G['clusters']}_{G['ll']}_{G['opts']}"
    )
    m = re.search(full_regex_pypsa_eur, spec)
    if m is not None:
        return m.groupdict()

    raise ValueError(f"Could not parse network space {spec}")


def parse_year_wildcard(w):
    """
    Parse a {year} wildcard to a list of years.

    The wildcard can be of the form `1980+1990+2000-2002`; a set of
    ranges (two years joined by a `-`) and individual years all
    separated by `+`s. The above wildcard is parsed to the list [1980,
    1990, 2000, 2001, 2002].
    """
    years = []
    for rng in w.split("+"):
        try:
            if "-" in rng:
                # `rng` is a range of years.
                [start, end] = rng.split("-")
                # Check that the range is well-formed.
                if end < start:
                    raise ValueError(f"Malformed range of years {rng}.")
                # Add the range (inclusive) to the set of years.
                years.extend(range(int(start), int(end) + 1))
            else:
                # `rng` is just a single year.
                years.append(int(rng))
        except ValueError:
            raise ValueError(f"Illegal range of years {rng} encountered.")
    # Sort the years before returning.
    return sorted(years)
