from pathlib import Path
import sys

def res(*parts: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS) / "aii" / "resources"
    else:
        base = Path(__file__).resolve().parent / "resources"
    return base.joinpath(*parts)
