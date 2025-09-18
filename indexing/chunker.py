import ast
from pathlib import Path
from typing import List, Optional, Tuple

from .models import CodeChunk


MAX_LINES = 250

def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _invert_ranges(total: int, covered: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not covered:
        return [(1, total)]
    covered = _merge_ranges(covered)
    gaps: List[Tuple[int, int]] = []
    cur = 1
    for s, e in covered:
        if cur < s:
            gaps.append((cur, s - 1))
        cur = e + 1
    if cur <= total:
        gaps.append((cur, total))
    return gaps


def _split_if_too_big(s: int, e: int, max_lines: int = 250) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    i = s
    while i <= e:
        j = min(e, i + max_lines - 1)
        out.append((i, j))
        if j == e:
            break
        i = j + 1
    return out


def _chunk_unknown_module_body(
    path: Path,
    tree: ast.AST,
    text: str,
    imports: List[str],
    header_end: int = 0
) -> List['CodeChunk']:
    lines = text.splitlines()
    total_lines = len(lines)
    covered: List[Tuple[int, int]] = []

    # 1. exclude imports, functions, classes
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s, e = node.lineno, node.end_lineno
            covered.append((s, e))

    # 2. exclude header/docstring region if provided
    if header_end > 0:
        covered.append((1, header_end))

    gaps = _invert_ranges(total_lines, covered)

    chunks: List[CodeChunk] = []
    for s, e in gaps:
        for ws, we in _split_if_too_big(s, e, max_lines=MAX_LINES):
            body_text = "\n".join(lines[ws - 1: we])
            if body_text.strip():
                chunks.append(
                    CodeChunk(
                        id=f"{path}::module_body:{ws}-{we}",
                        file_path=str(path),
                        symbol_name="__module_body__",
                        symbol_type="module_body",
                        start_line=ws,
                        end_line=we,
                        text=body_text,
                        docstring=None,
                        imports=imports,
                    )
                )
    return chunks


def _read_file_text(path: Path) -> str:
    """Load file contents with utf-8 and replacement for bad bytes."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _leading_comments(text: str) -> Optional[str]:
    """Capture top-of-file comments and blank lines prior to the first statement."""
    lines = text.splitlines()
    collected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            collected.append(line)
        else:
            break
    out = "\n".join(collected).strip()
    return out if out else None


    


def _extract_imports(tree: ast.AST) -> List[str]:
    """Collect import and from-import targets for context metadata."""
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return imports


def chunk_python_file(path: Path) -> List[CodeChunk]:
    text = _read_file_text(path)
    chunks: List[CodeChunk] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunks

    file_rel = str(path)
    imports = _extract_imports(tree)
    module_doc = ast.get_docstring(tree)
    header_comments = _leading_comments(text)

    file_text_preview_parts: List[str] = []
    if header_comments:
        file_text_preview_parts.append(header_comments)
    if module_doc:
        file_text_preview_parts.append(module_doc)
    file_text_preview = "\n\n".join(file_text_preview_parts).strip()

    if file_text_preview:
        chunks.append(
            CodeChunk(
                id=f"{file_rel}::module",
                file_path=file_rel,
                symbol_name=Path(file_rel).stem,
                symbol_type="module",
                start_line=1,
                end_line=file_text_preview.count("\n") + 1,
                text=file_text_preview,
                docstring=module_doc,
                imports=imports,
            )
        )

    # nodes = [node for node in tree.body if not (isinstance(node))]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno
            node_text = "\n".join(text.splitlines()[start - 1 : end])
            doc = ast.get_docstring(node)
            chunks.append(
                CodeChunk(
                    id=f"{file_rel}::func::{node.name}:{start}-{end}",
                    file_path=file_rel,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=start,
                    end_line=end,
                    text=node_text,
                    docstring=doc,
                    imports=imports,
                )
            )
        elif isinstance(node, ast.ClassDef):
            c_start = node.lineno
            c_end = node.end_lineno
            class_text = "\n".join(text.splitlines()[c_start - 1 : c_end])
            c_doc = ast.get_docstring(node)
            chunks.append(
                CodeChunk(
                    id=f"{file_rel}::class::{node.name}:{c_start}-{c_end}",
                    file_path=file_rel,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=c_start,
                    end_line=c_end,
                    text=class_text,
                    docstring=c_doc,
                    imports=imports,
                )
            )
    header_end = file_text_preview.count("\n") + 1
    chunks.extend(_chunk_unknown_module_body(path, tree, text, imports, header_end))
    return chunks
