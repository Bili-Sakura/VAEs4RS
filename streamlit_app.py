import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st
from diffusers.utils import make_image_grid
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from config import DATASET_CONFIGS, VAE_CONFIGS
except Exception:
    DATASET_CONFIGS = {}
    VAE_CONFIGS = {}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    if path.exists():
        return path.resolve()
    return candidate


def extract_original_stem(saved_filename: str) -> str:
    stem = Path(saved_filename).stem
    if "_batch" in stem:
        return stem.rsplit("_batch", 1)[0]
    return stem


def order_models(models: List[str]) -> List[str]:
    if not VAE_CONFIGS:
        return sorted(models)
    ordered = [name for name in VAE_CONFIGS.keys() if name in models]
    extras = sorted([name for name in models if name not in ordered])
    return ordered + extras


def order_datasets(datasets: List[str]) -> List[str]:
    if not DATASET_CONFIGS:
        return sorted(datasets)
    ordered = [name for name in DATASET_CONFIGS.keys() if name in datasets]
    extras = sorted([name for name in datasets if name not in ordered])
    return ordered + extras


@st.cache_data
def find_datasets(results_dir_str: str) -> Dict[str, str]:
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        return {}
    datasets = {}
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        original_dir = child / "images" / "original"
        if original_dir.exists():
            datasets[child.name] = str(original_dir)
    return datasets


@st.cache_data
def find_models(results_dir_str: str, dataset_name: str) -> List[str]:
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        return []
    models = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        recon_dir = child / dataset_name / "images" / "reconstructed"
        if recon_dir.exists():
            models.append(child.name)
    return order_models(models)


@st.cache_data
def list_saved_images(images_dir_str: str) -> List[str]:
    images_dir = Path(images_dir_str)
    if not images_dir.exists():
        return []
    return sorted([path.name for path in images_dir.glob("*.png")])


@st.cache_data
def list_classes(dataset_root_str: str) -> List[str]:
    dataset_root = Path(dataset_root_str)
    if not dataset_root.exists():
        return []
    return sorted([path.name for path in dataset_root.iterdir() if path.is_dir()])


@st.cache_data
def get_class_stems(dataset_root_str: str, class_name: str) -> Set[str]:
    dataset_root = Path(dataset_root_str)
    class_dir = dataset_root / class_name
    if not class_dir.exists():
        return set()
    stems = set()
    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(file_path.stem)
    return stems


@st.cache_data
def list_reconstructed_filenames(recon_dir_str: str) -> Set[str]:
    recon_dir = Path(recon_dir_str)
    if not recon_dir.exists():
        return set()
    return {path.name for path in recon_dir.glob("*.png")}


def load_images(images_dir: Path, filenames: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for name in filenames:
        image_path = images_dir / name
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            images.append(img.convert("RGB"))
    return images


def build_image_grid(images: List[Image.Image], columns: int) -> Optional[Image.Image]:
    if not images:
        return None
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    return make_image_grid(images, rows=rows, cols=columns)


def main() -> None:
    st.set_page_config(page_title="VAE Baseline Viewer", layout="wide")
    st.title("VAE Baseline Reconstruction Viewer")

    with st.sidebar:
        st.header("Controls")
        results_dir_input = st.text_input(
            "Results directory",
            value="datasets/BiliSakura/VAEs4RS",
        )
        resolved_results_dir = resolve_path(results_dir_input)
        st.caption(f"Resolved results path: {resolved_results_dir}")

        datasets_map = find_datasets(str(resolved_results_dir))
        if not datasets_map:
            st.error("No datasets found in results directory.")
            st.stop()

        dataset_name = st.selectbox(
            "Dataset",
            options=order_datasets(list(datasets_map.keys())),
        )

        dataset_root = None
        if dataset_name in DATASET_CONFIGS:
            dataset_root = resolve_path(DATASET_CONFIGS[dataset_name].root)
        dataset_root_input = st.text_input(
            "Dataset root (for class filter)",
            value=str(dataset_root) if dataset_root else "",
        )
        dataset_root = resolve_path(dataset_root_input) if dataset_root_input else None

        class_names: List[str] = []
        if dataset_root and dataset_root.exists():
            class_names = list_classes(str(dataset_root))
        class_options = ["All"] + class_names
        class_choice = st.selectbox("Class category", options=class_options)

        models = find_models(str(resolved_results_dir), dataset_name)
        if not models:
            st.error("No models found for the selected dataset.")
            st.stop()
        model_selection = st.multiselect(
            "Models",
            options=models,
            default=models,
        )

        sample_count = st.number_input(
            "Number of samples",
            min_value=1,
            max_value=64,
            value=8,
            step=1,
        )
        random_button = st.button("Random display", type="primary")

    original_dir = Path(datasets_map[dataset_name])
    available_files = list_saved_images(str(original_dir))
    if not available_files:
        st.error("No saved original images found for this dataset.")
        st.stop()

    if class_choice != "All" and dataset_root and dataset_root.exists():
        class_stems = get_class_stems(str(dataset_root), class_choice)
        if class_stems:
            available_files = [
                name for name in available_files
                if extract_original_stem(name) in class_stems
            ]
        else:
            available_files = []

    if not available_files:
        st.error("No images matched the selected class category.")
        st.stop()

    selected_models = model_selection if model_selection else []
    for model_name in selected_models:
        recon_dir = resolved_results_dir / model_name / dataset_name / "images" / "reconstructed"
        recon_files = list_reconstructed_filenames(str(recon_dir))
        if recon_files:
            available_files = [name for name in available_files if name in recon_files]
        else:
            available_files = []
        if not available_files:
            break

    if not available_files:
        st.error("No images matched across selected models.")
        st.stop()

    selection_signature = (
        str(resolved_results_dir),
        dataset_name,
        class_choice,
        tuple(selected_models),
        int(sample_count),
    )
    if st.session_state.get("selection_signature") != selection_signature:
        st.session_state["selection_signature"] = selection_signature
        st.session_state["sampled_filenames"] = []

    if random_button or not st.session_state.get("sampled_filenames"):
        sample_size = min(int(sample_count), len(available_files))
        st.session_state["sampled_filenames"] = random.sample(available_files, sample_size)

    sampled_filenames = st.session_state["sampled_filenames"]
    grid_columns = min(6, len(sampled_filenames)) if sampled_filenames else 1

    original_images = load_images(original_dir, sampled_filenames)
    original_grid = build_image_grid(original_images, grid_columns)
    if original_grid:
        st.image(original_grid, use_container_width=True)
        st.caption(f"Ground truth ({dataset_name} | {class_choice})")

    for model_name in selected_models:
        recon_dir = resolved_results_dir / model_name / dataset_name / "images" / "reconstructed"
        recon_images = load_images(recon_dir, sampled_filenames)
        recon_grid = build_image_grid(recon_images, grid_columns)
        if recon_grid:
            st.image(recon_grid, use_container_width=True)
            st.caption(model_name)


if __name__ == "__main__":
    main()
