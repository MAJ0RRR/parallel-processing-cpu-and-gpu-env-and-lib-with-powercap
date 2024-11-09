import os
import json
import re

from dataclasses import dataclass, asdict, fields


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d
    

@dataclass
class RunParameters:
    app_name: str
    cpu_enabled: bool
    number_of_streams: int
    number_od_nodes: int
    batch_size: int
    powercap: int | None
    problem_size: int


@dataclass
class SingleRunResult:
    execution_duration: float # in seconds
    energy_used: float # in Watts

    @staticmethod
    def from_output(stdout: str, stderr: str) -> "SingleRunResult":
        return SingleRunResult(
            execution_duration=SingleRunResult.get_execution_duration_from_output(stdout=stdout, stderr=stderr), 
            energy_used=SingleRunResult.get_energy_used_from_output(stdout=stdout, stderr=stderr),
        )

    @staticmethod
    def get_execution_duration_from_output(stdout: str, stderr: str):
        main_time_match = re.search(r'Main elapsed time=([\d.]+)', stderr)
        return float(main_time_match.group(1))

    @staticmethod
    def get_energy_used_from_output(stdout: str, stderr: str):
        return 0.0
    

@dataclass
class ExperimentResult:
    description: str
    parameters: RunParameters
    runs: list[SingleRunResult]
    
    def to_file(self, file_path: str | os.PathLike):
        with open(file_path, "w") as file:
            json.dump(asdict(self), file, indent=4)

    @classmethod
    def from_file(cls, file_path: str | os.PathLike) -> "ExperimentResult":
        with open(file_path, "r") as file:
            data = json.load(file)
        return dataclass_from_dict(cls, data)
    