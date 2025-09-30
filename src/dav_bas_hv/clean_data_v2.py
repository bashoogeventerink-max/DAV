# import packages
from pathlib import Path
import pandas as pd
from datetime import datetime
import tomllib

# Load the data from the data folder
configfile = Path("../config.toml").resolve()
with configfile.open("rb") as f:
    config = tomllib.load(f)
processed = Path("../data/processed")
datafile = processed / config["inputpath"]
if not datafile.exists():
    logger.warning(
        f"{datafile} does not exist. Maybe first run src/preprocess.py, or check the timestamp!"
    )

print(datafile)