from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

ASSETS_PATH = ROOT_DIR / "assets"
FIGS_PATH = ASSETS_PATH / "figs"

DATA_DIR = ROOT_DIR / "data"
MNIST_PATH = DATA_DIR / "MNIST.npz"