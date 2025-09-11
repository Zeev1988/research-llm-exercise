import argparse
from pathlib import Path

from indexing.indexer import index_repository


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="research-llm-exercise CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build an index for a repository")
    p_index.add_argument("repo", type=str, help="Path to local repository root")
    p_index.add_argument("--out", type=str, default=None, help="Output directory for the index")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    # args = _parse_args(argv)
    # if args.command == "index":
    #     repo = Path(args.repo)
    #     out = Path(args.out).resolve() if args.out else None
    index_repository(Path(r'/Users/zeevhananis/Downloads/signify-master'),
                     output_dir=Path(r'/Users/zeevhananis/Downloads/res').resolve())


if __name__ == "__main__":
    main()


