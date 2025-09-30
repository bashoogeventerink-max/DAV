# To easily create data objects with minimal setup
from dataclasses import dataclass 

# Import path to easily track and understand the filesystem patch
from pathlib import Path

# Give option to say variable can be None:
from typing import Optional

# Define a class 
from pydantic import BaseModel

HOUR = 60 * 60
DAY = HOUR * 24

class BaseRegexes(BaseModel):
    timestamp: str
    author: str
    message: str


# Automaticcaly generate __init__ method:
@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path