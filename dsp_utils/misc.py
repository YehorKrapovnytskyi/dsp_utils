from typing import Iterator, Tuple, Callable
from pathlib import Path


# --- Reusable path generator for feature extraction ---
def stream_pipeline_paths(
    raw_dir: Path, 
    features_dir: Path, 
    extension: str
) -> Iterator[Tuple[Path, Path]]:
    """Yields (source_file_path, target_directory_path) pairs while replicating hierarchy."""
    if not raw_dir.is_dir():
        raise ValueError(f"Raw data directory does not exist: {raw_dir}")
        
    target_suffix = extension.lower()
    
    for dirpath, _, filenames in raw_dir.walk():
        target_files = [f for f in filenames if f.lower().endswith(target_suffix)]
        
        if not target_files:
            continue
            
        target_dir = features_dir / dirpath.relative_to(raw_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_name in target_files:
            yield dirpath / file_name, target_dir


# --- Reusable path generator for augmentation ---
def stream_augmentation_pipeline(
    raw_dir: Path, 
    augmented_dir: Path, 
    strategies: dict, 
    extension: str
) -> Iterator[Tuple[Path, Path, str, Callable]]:
    """
    Traverses directories lazily, generates inverted hierarchy trees per strategy,
    and yields (src_file_path, target_dir_path, strategy_name, transformation_function).
    """
    if not raw_dir.is_dir():
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return
        
    target_suffix = extension.lower()
    
    for dirpath, _, filenames in raw_dir.walk():
        target_files = [f for f in filenames if f.lower().endswith(target_suffix)]
        
        if not target_files:
            continue
            
        rel_path = dirpath.relative_to(raw_dir)
        
        # Pre-create folders for all strategies before yielding files
        for aug_name, aug_func in strategies.items():
            target_dir = augmented_dir / aug_name / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for file_name in target_files:
                yield dirpath / file_name, target_dir, aug_name, aug_func