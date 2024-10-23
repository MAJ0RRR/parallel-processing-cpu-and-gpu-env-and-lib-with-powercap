from dataclasses import dataclass

@dataclass
class RunParameters:
    app_name: str
    cpu_enabled: bool
    number_of_streams: int
    batch_size: int
    powercap: int | None
    problem_size: int