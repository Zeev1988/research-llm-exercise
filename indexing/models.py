from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeChunk:
    id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    text: str
    docstring: Optional[str]
    imports: List[str]


