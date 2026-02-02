import io
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


def add_ground_truth_border(grid_image: Image.Image, num_cols: int, border_width: int = 3) -> Image.Image:
    """Add a red vertical border rectangle on the ground truth column (leftmost column)."""
    from PIL import ImageDraw
    img_copy = grid_image.copy()
    width, height = img_copy.size
    draw = ImageDraw.Draw(img_copy)
    
    # Calculate the width of each column
    col_width = width // num_cols
    
    # Draw red vertical rectangle on the leftmost column (ground truth)
    # Rectangle spans from top to bottom
    draw.rectangle(
        [(0, 0), (col_width - 1, height - 1)],
        outline="red",
        width=border_width
    )
    
    return img_copy


def build_image_grid(images: List[Image.Image], columns: int) -> Optional[Image.Image]:
    if not images:
        return None
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    required_count = rows * columns
    # Pad images list to match the required grid size
    padded_images = list(images)
    if len(padded_images) < required_count:
        # Pad with copies of the last image
        last_image = padded_images[-1] if padded_images else None
        if last_image:
            padded_images.extend([last_image.copy() for _ in range(required_count - len(padded_images))])
    return make_image_grid(padded_images, rows=rows, cols=columns)


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
    
    # Load all images: ground truth + all model reconstructions
    original_images = load_images(original_dir, sampled_filenames)
    model_images_dict = {}
    for model_name in selected_models:
        recon_dir = resolved_results_dir / model_name / dataset_name / "images" / "reconstructed"
        recon_images = load_images(recon_dir, sampled_filenames)
        model_images_dict[model_name] = recon_images
    
    # Build combined grid: rows = samples, columns = [Ground Truth, Model1, Model2, ...]
    num_samples = len(sampled_filenames)
    num_models = len(selected_models)
    num_cols = 1 + num_models  # 1 for ground truth + models
    
    if num_samples > 0 and num_cols > 0:
        # Create a blank white image for padding if needed
        blank_image = None
        if original_images:
            blank_image = Image.new("RGB", original_images[0].size, color="white")
        
        # Arrange images row by row: each row is [gt, model1, model2, ...]
        combined_images = []
        for i in range(num_samples):
            # Add ground truth image
            if i < len(original_images):
                combined_images.append(original_images[i])
            elif original_images:
                combined_images.append(original_images[-1].copy())
            elif blank_image:
                combined_images.append(blank_image.copy())
            
            # Add model images for this sample
            for model_name in selected_models:
                model_images = model_images_dict.get(model_name, [])
                if i < len(model_images):
                    combined_images.append(model_images[i])
                elif model_images:
                    combined_images.append(model_images[-1].copy())
                elif blank_image:
                    combined_images.append(blank_image.copy())
        
        # Ensure we have exactly rows * cols images
        required_count = num_samples * num_cols
        if len(combined_images) < required_count and combined_images:
            # Pad with copies of the last image
            last_image = combined_images[-1]
            combined_images.extend([last_image.copy() for _ in range(required_count - len(combined_images))])
        elif len(combined_images) > required_count:
            # Trim if somehow we have too many
            combined_images = combined_images[:required_count]
        
        if combined_images:
            combined_grid = make_image_grid(combined_images, rows=num_samples, cols=num_cols)
            # Add red border on the ground truth column (leftmost column)
            combined_grid = add_ground_truth_border(combined_grid, num_cols)
            st.image(combined_grid, use_container_width=True)
            
            # Create caption with column headers
            headers = ["Ground Truth"] + list(selected_models)
            st.caption(f"Columns: {' | '.join(headers)} | Rows: {num_samples} samples ({dataset_name} | {class_choice})")
            
            # Save buttons
            col1, col2 = st.columns(2)
            
            # Generate filename
            filename_base = f"{dataset_name}_{class_choice}_{'_'.join(selected_models)}_{num_samples}samples"
            
            # PNG download
            png_buffer = io.BytesIO()
            combined_grid.save(png_buffer, format="PNG")
            png_buffer.seek(0)
            col1.download_button(
                label="Save as PNG",
                data=png_buffer,
                file_name=f"{filename_base}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # PDF download
            try:
                import img2pdf
                pdf_buffer = io.BytesIO()
                # Convert to RGB if needed
                pdf_image = combined_grid.convert("RGB")
                # Save image to bytes first
                img_buffer = io.BytesIO()
                pdf_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                # Convert to PDF using img2pdf
                pdf_bytes = img2pdf.convert(img_buffer)
                pdf_buffer.write(pdf_bytes)
                pdf_buffer.seek(0)
                col2.download_button(
                    label="Save as PDF",
                    data=pdf_buffer,
                    file_name=f"{filename_base}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except ImportError:
                # Fallback: use PIL's PDF support (may not work on all systems)
                pdf_buffer = io.BytesIO()
                pdf_image = combined_grid.convert("RGB")
                try:
                    pdf_image.save(pdf_buffer, format="PDF")
                    pdf_buffer.seek(0)
                    col2.download_button(
                        label="Save as PDF",
                        data=pdf_buffer,
                        file_name=f"{filename_base}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception:
                    col2.warning("PDF export requires img2pdf library. Install with: pip install img2pdf")


if __name__ == "__main__":
    main()
