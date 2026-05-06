import numpy as np
import zipfile
import matplotlib.pyplot as plt
from face_id.data import preprocess_image

def load_pgm(filename):
    """
    Load a PGM image file (P5 format, binary grayscale).
    
    Args:
        filename (str): Path to the PGM file, or 'archive.zip/internal/path/file.pgm' for files inside zip archives.
    
    Returns:
        numpy.ndarray: 2D array representing the grayscale image.
    """
    if '.zip' in filename:
        zip_path, internal_path = filename.split('.zip', 1)
        zip_path += '.zip'
        internal_path = internal_path.lstrip('/')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_path) as f:
                # Read magic number
                magic = f.readline().strip()
                if magic != b'P5':
                    raise ValueError("Not a valid P5 PGM file")
                
                # Skip comments
                while True:
                    pos = f.tell()
                    line = f.readline().strip()
                    if not line.startswith(b'#'):
                        f.seek(pos)  # Go back to before this line
                        break
                
                # Read width and height
                line = f.readline().strip()
                width, height = map(int, line.split())
                
                # Read maxval
                maxval = int(f.readline().strip())
                
                # Read image data
                if maxval < 256:
                    dtype = np.uint8
                else:
                    dtype = np.uint16
                
                data = np.frombuffer(f.read(), dtype=dtype)
                image = data.reshape((height, width))
                
                return image
    else:
        with open(filename, 'rb') as f:
            # Read magic number
            magic = f.readline().strip()
            if magic != b'P5':
                raise ValueError("Not a valid P5 PGM file")
            
            # Skip comments
            while True:
                pos = f.tell()
                line = f.readline().strip()
                if not line.startswith(b'#'):
                    f.seek(pos)  # Go back to before this line
                    break
            
            # Read width and height
            line = f.readline().strip()
            width, height = map(int, line.split())
            
            # Read maxval
            maxval = int(f.readline().strip())
            
            # Read image data
            if maxval < 256:
                dtype = np.uint8
            else:
                dtype = np.uint16
            
            data = np.frombuffer(f.read(), dtype=dtype)
            image = data.reshape((height, width))
            
            return image

def load_pgm_vector(filename):
    img = load_pgm(filename)
    vector = img.flatten().astype(np.float32)
    min_val = vector.min()
    max_val = vector.max()
    if max_val > min_val:
        vector = ((vector - min_val) / (max_val - min_val))*2 - 1
    else:
        raise ValueError("Image has no variation in pixel values.")
    return vector

def get_file(p_num, i_num):
    """
    Get the file path for a specific person and image number from the train set zip.

    Args:
        p_num (int): Person number (e.g., 27).
        i_num (int): Image number (e.g., 447).

    Returns:
        str: The full path including zip, e.g., 'zip_name/internal/path/file.pgm'.
    """
    from pathlib import Path
    zip_name = Path(__file__).resolve().parent.parent / "Train Set (Labeled)-20260405T164823Z-3-001.zip"
    internal_path = f"Train Set (Labeled)/p{p_num}_i{i_num}.pgm"
    return f"{zip_name}/{internal_path}"


if __name__ == "__main__":

    # (person_num, image_num) samples to display
    SAMPLES = [(24, 0), (27, 0), (5, 0), (10, 0)]
    n = len(SAMPLES)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Before vs After Preprocessing", fontsize=16, fontweight="bold", y=1.01)

    for col, (p_num, i_num) in enumerate(SAMPLES):
        raw = load_pgm(get_file(p_num, i_num))
        processed = preprocess_image(raw)

        # Top row — original
        axes[0, col].imshow(raw, cmap="gray")
        axes[0, col].set_title(f"p{p_num}_i{i_num}\n(original {raw.shape[1]}×{raw.shape[0]})", fontsize=10)
        axes[0, col].axis("off")

        # Bottom row — preprocessed
        axes[1, col].imshow(processed, cmap="gray")
        axes[1, col].set_title(
            f"preprocessed {processed.shape[1]}×{processed.shape[0]}\n"
            f"mean={processed.mean():.2f}  std={processed.std():.2f}",
            fontsize=9,
        )
        axes[1, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("Original", fontsize=13, labelpad=10)
    axes[1, 0].set_ylabel("Preprocessed", fontsize=13, labelpad=10)

    plt.tight_layout()
    plt.show()
