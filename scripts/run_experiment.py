from pathlib import Path

from sigver.utils.config import load_config


def main() -> None:
    config_path = Path("configs/experiments/exp01_siamese_resnet_cedar.yaml")
    config = load_config(config_path)
    print(config)


if __name__ == "__main__":
    main()
