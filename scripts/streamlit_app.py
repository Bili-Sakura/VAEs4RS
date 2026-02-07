#!/usr/bin/env python3
"""
VAE Baseline Reconstruction Viewer - Streamlit App

Usage:
    streamlit run scripts/streamlit_app.py
"""

import io
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st
from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.config import get_config
    cfg = get_config()
    DATASET_CONFIGS = cfg.datasets
    VAE_CONFIGS = cfg.vaes
except Exception:
    DATASET_CONFIGS = {}
    VAE_CONFIGS = {}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidate = (PROJECT_ROOT / path).resolve()
    return candidate if candidate.exists() else path.resolve()


def extract_original_stem(saved_filename: str) -> str:
    stem = Path(saved_filename).stem
    return stem.rsplit("_batch", 1)[0] if "_batch" in stem else stem


def order_items(items: List[str], config_dict: Dict) -> List[str]:
    if not config_dict:
        return sorted(items)
    ordered = [name for name in config_dict.keys() if name in items]
    extras = sorted([name for name in items if name not in ordered])
    return ordered + extras


@st.cache_data
def find_datasets(results_dir_str: str) -> Dict[str, str]:
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        return {}
    return {
        child.name: str(child / "images" / "original")
        for child in results_dir.iterdir()
        if child.is_dir() and (child / "images" / "original").exists()
    }


@st.cache_data
def find_models(results_dir_str: str, dataset_name: str) -> List[str]:
    results_dir = Path(results_dir_str)
    if not results_dir.exists():
        return []
    models = [
        child.name for child in results_dir.iterdir()
        if child.is_dir() and (child / dataset_name / "images" / "reconstructed").exists()
    ]
    return order_items(models, VAE_CONFIGS)


@st.cache_data
def list_saved_images(images_dir_str: str) -> List[str]:
    images_dir = Path(images_dir_str)
    return sorted([p.name for p in images_dir.glob("*.png")]) if images_dir.exists() else []


@st.cache_data
def list_classes(dataset_root_str: str) -> List[str]:
    dataset_root = Path(dataset_root_str)
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()]) if dataset_root.exists() else []


@st.cache_data
def get_class_stems(dataset_root_str: str, class_name: str) -> Set[str]:
    class_dir = Path(dataset_root_str) / class_name
    if not class_dir.exists():
        return set()
    return {p.stem for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


@st.cache_data
def list_reconstructed_filenames(recon_dir_str: str) -> Set[str]:
    recon_dir = Path(recon_dir_str)
    return {p.name for p in recon_dir.glob("*.png")} if recon_dir.exists() else set()


def load_images(images_dir: Path, filenames: List[str]) -> List[Image.Image]:
    images = []
    for name in filenames:
        path = images_dir / name
        if path.exists():
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
    return images


def add_border(grid_image: Image.Image, num_cols: int, border_width: int = 3) -> Image.Image:
    """Add red border on ground truth column."""
    img_copy = grid_image.copy()
    draw = ImageDraw.Draw(img_copy)
    col_width = img_copy.width // num_cols
    draw.rectangle([(0, 0), (col_width - 1, img_copy.height - 1)], outline="red", width=border_width)
    return img_copy


def main():
    st.set_page_config(page_title="VAE Baseline Viewer", layout="wide")
    st.title("VAE Baseline Reconstruction Viewer")

    with st.sidebar:
        st.header("Controls")
        results_dir_input = st.text_input("Results directory", value="outputs")
        resolved_results_dir = resolve_path(results_dir_input)
        st.caption(f"Path: {resolved_results_dir}")

        datasets_map = find_datasets(str(resolved_results_dir))
        if not datasets_map:
            st.error("No datasets found.")
            st.stop()

        dataset_name = st.selectbox("Dataset", options=order_items(list(datasets_map.keys()), DATASET_CONFIGS))

        # Dataset root for class filtering
        dataset_root = None
        if dataset_name in DATASET_CONFIGS:
            dataset_root = resolve_path(DATASET_CONFIGS[dataset_name].root)
        dataset_root_input = st.text_input("Dataset root", value=str(dataset_root) if dataset_root else "")
        dataset_root = resolve_path(dataset_root_input) if dataset_root_input else None

        class_names = list_classes(str(dataset_root)) if dataset_root and dataset_root.exists() else []
        class_choice = st.selectbox("Class", options=["All"] + class_names)

        models = find_models(str(resolved_results_dir), dataset_name)
        if not models:
            st.error("No models found.")
            st.stop()
        model_selection = st.multiselect("Models", options=models, default=models)

        sample_count = st.number_input("Samples", min_value=1, max_value=64, value=8)
        random_button = st.button("Random", type="primary")

    # Get available files
    original_dir = Path(datasets_map[dataset_name])
    available_files = list_saved_images(str(original_dir))
    if not available_files:
        st.error("No images found.")
        st.stop()

    # Filter by class
    if class_choice != "All" and dataset_root and dataset_root.exists():
        class_stems = get_class_stems(str(dataset_root), class_choice)
        available_files = [f for f in available_files if extract_original_stem(f) in class_stems]

    # Filter by model availability
    for model_name in model_selection or []:
        recon_dir = resolved_results_dir / model_name / dataset_name / "images" / "reconstructed"
        recon_files = list_reconstructed_filenames(str(recon_dir))
        available_files = [f for f in available_files if f in recon_files]

    if not available_files:
        st.error("No matching images.")
        st.stop()

    # Session state for sampling
    sig = (str(resolved_results_dir), dataset_name, class_choice, tuple(model_selection or []), int(sample_count))
    if st.session_state.get("sig") != sig:
        st.session_state["sig"] = sig
        st.session_state["samples"] = []

    if random_button or not st.session_state.get("samples"):
        st.session_state["samples"] = random.sample(available_files, min(int(sample_count), len(available_files)))

    sampled = st.session_state["samples"]
    
    # Load images
    original_images = load_images(original_dir, sampled)
    model_images = {
        m: load_images(resolved_results_dir / m / dataset_name / "images" / "reconstructed", sampled)
        for m in (model_selection or [])
    }
    
    # Build grid
    num_samples = len(sampled)
    num_cols = 1 + len(model_selection or [])
    
    if num_samples > 0 and original_images:
        combined = []
        for i in range(num_samples):
            combined.append(original_images[i] if i < len(original_images) else original_images[-1].copy())
            for m in (model_selection or []):
                imgs = model_images.get(m, [])
                combined.append(imgs[i] if i < len(imgs) else (imgs[-1].copy() if imgs else original_images[-1].copy()))
        
        grid = make_image_grid(combined, rows=num_samples, cols=num_cols)
        grid = add_border(grid, num_cols)
        st.image(grid, use_container_width=True)
        
        headers = ["Ground Truth"] + (model_selection or [])
        st.caption(f"Columns: {' | '.join(headers)}")
        
        # Download buttons
        col1, col2 = st.columns(2)
        filename = f"{dataset_name}_{class_choice}_{num_samples}samples"
        
        buf = io.BytesIO()
        grid.save(buf, format="PNG")
        buf.seek(0)
        col1.download_button("Save PNG", data=buf, file_name=f"{filename}.png", mime="image/png", use_container_width=True)
        
        try:
            import img2pdf
            pdf_buf = io.BytesIO()
            img_buf = io.BytesIO()
            grid.convert("RGB").save(img_buf, format="PNG")
            img_buf.seek(0)
            pdf_buf.write(img2pdf.convert(img_buf))
            pdf_buf.seek(0)
            col2.download_button("Save PDF", data=pdf_buf, file_name=f"{filename}.pdf", mime="application/pdf", use_container_width=True)
        except ImportError:
            col2.info("Install img2pdf for PDF export")


if __name__ == "__main__":
    main()
