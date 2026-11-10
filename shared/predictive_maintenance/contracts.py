from typing import Literal

OperatingMode = Literal["high load", "normal load", "idle"]
Priority = Literal["high", "medium", "low"]
MaintenanceAction = Literal[
    "replace bearing",
    "repair cooling system",
    "replace valve",
    "align motor",
]
