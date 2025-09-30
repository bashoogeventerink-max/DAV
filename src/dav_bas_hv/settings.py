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

# Apply date formats per software type

iosRegexes = BaseRegexes(
    timestamp=r"\[(.+?)]\s.+?:.+",
    author=r"\[.+?]\s(.+?):.+",
    message=r"\[.+?]\s.+?:(.+)",
)

androidRegexes = BaseRegexes(
    timestamp=r"(.+?)\s-\s.+?:.+",
    author=r".+?\s-\s(.+?):.+",
    message=r".+?\s-\s.+?:(.*)",
)

oldRegexes = BaseRegexes(
    timestamp=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}",
    author=r"(?<=\s-\s)(.*?)(?=:)",
    message=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}[-~a-zA-Z0-9\s]+:",
)

csvRegexes = BaseRegexes(
    timestamp=r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    author=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},([^,]+),",
    message=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},[^,]+,(.+)",
)

# Automaticcaly generate __init__ method:
@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path

# Preprocess settings to format date, drop authors, apply regex
class PreprocessConfig(BaseModel):
    folders: Folders
    regexes: BaseRegexes
    datetime_format: str
    drop_authors: list[str] = []