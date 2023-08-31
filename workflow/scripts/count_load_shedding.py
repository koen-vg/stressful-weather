# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output the load shedding that we get from running a network based on year {op_year} with the weather of year {weather_year}."""

import pypsa
from _helpers import configure_logging

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)
    # Note that the operations folder already exists.

    # Load the input network.
    n = pypsa.Network(snakemake.input.network)

    # Save the load shedding in a dataframe.
    load_shedding = n.generators_t.p.filter(like="load shedding", axis=1)

    # Re-index the dataframe to the weather year and remove possible leap days.
    # weather_year = snakemake.wildcards["weather_year"]
    # new_index = pd.DatetimeIndex(pd.date_range(start=f"{weather_year-1}-07-01", end=f"{weather_year}-07-01", freq="H", closed="left"))
    # new_index = new_index[~((new_index.month == 2) & (new_index.day == 29))]
    # load_shedding.index = new_index

    # Export the results.
    load_shedding.to_csv(snakemake.output[0])
