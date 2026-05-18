from dataclasses import dataclass


@dataclass(frozen=True)
class ScanOptions:
    """Runtime options for a scan run."""

    marker_profile: str = "auto_heur"  # auto_heur | auto_ai | numeric | symbol | letter
    allow_ai_pairing: bool = True
    ai_debug: bool = False
