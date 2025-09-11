import argparse
from pathlib import Path
from indexing.indexer import index_repository

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Index a repository for code QA")
    parser.add_argument("--repo", type=str, required=True, help="Path to local repository root")
    parser.add_argument("--out", type=str, default=None, help="Output directory for the index")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    repo = Path(args.repo)
    out = Path(args.out).resolve() if args.out else None
    index_repository(repo, out)


if __name__ == "__main__":
    main()
