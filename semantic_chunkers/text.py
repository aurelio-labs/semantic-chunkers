from typing import Union

from colorama import Fore
from semantic_chunkers.chunkers.consecutive_sim import ConsecutiveSimSplitter
from semantic_chunkers.chunkers.cumulative_sim import CumulativeSimSplitter

# Define a type alias for the splitter to simplify the annotation
SplitterType = Union[ConsecutiveSimSplitter, CumulativeSimSplitter, None]

colors = [
    Fore.WHITE,
    Fore.RED,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.MAGENTA,
    Fore.CYAN,
]