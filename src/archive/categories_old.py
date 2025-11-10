# Import packages

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
import tomllib

warnings.simplefilter(action="ignore", category=FutureWarning)

# Get the path of the current script file
script_path = Path(__file__).resolve()

# Set the project root, which is one level up from the script's directory
# This assumes the script is in 'DAV/src' and the project root is 'DAV'
project_root = script_path.parent.parent.parent

# Load config 
configfile = project_root / "config.toml"
with configfile.open("rb") as f:
    config = tomllib.load(f)

# Load data
datafile = (project_root / Path(config["processed"]) / config["current"]).resolve()
if not datafile.exists():
    logger.warning(
        "Datafile does not exist. First run src/preprocess.py, and check the timestamp!"
    )
df = pd.read_parquet(datafile)
df.head()