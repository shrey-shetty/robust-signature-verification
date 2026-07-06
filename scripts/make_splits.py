from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    splits_dir = root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "README.md").write_text("Split files will be generated here.\n", encoding="utf-8")
    print(f"Created split directory at {splits_dir}")


if __name__ == "__main__":
    main()
