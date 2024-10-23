import os
import json

from dataclasses import dataclass, asdict

from cudampi_dataclasses import SingleRunResult, RunParameters

@dataclass
class ExperimentResult:
    description: str
    parameters: RunParameters
    runs: list[SingleRunResult]
    
    def to_file(self, file_path: str | os.PathLike):
        person_json = json.dumps(asdict(self))
        print(person_json)