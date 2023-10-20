from dataclasses import dataclass
from pathlib import Path

#This class is use to create entity just like configBox
@dataclass(frozen=True) #This is use to enable the entity 
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path