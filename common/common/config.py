from pathlib import Path

# PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

ASSETS_DIR = ROOT_DIR / "assets"
FIGS_DIR = ASSETS_DIR / "figs"

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
MNIST_RAW_PATH = DATA_RAW_DIR / "MNIST.npz"

DATA_PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MNIST_PREPROCESSED_PATH = DATA_PREPROCESSED_DIR / "MNIST_preprocessed.npz"

DATA_WEIGHTS_DIR = DATA_DIR / "weights"

# SEED
SEED = 42