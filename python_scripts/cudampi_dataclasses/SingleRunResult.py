import re

from dataclasses import dataclass



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