import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
import cv2
from pathlib import Path
import kornia
from typing import Optional, Tuple


def read_homography(path: Path) -> np.ndarray:
    """Read a homography matrix from a text file."""
    with open(path) as f:
        result = []
        for line in f.readlines():
            line = line.replace("  ", " ").strip()  # Normalize spaces
            elements = [e for e in line.split(" ") if e]
            if elements:
                result.append(elements)
        return np.array(result).astype(float)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale."""
    if not path.exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]  # BGR to RGB
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float32)


def load_image(path: Path, grayscale: bool = False) -> torch.Tensor:
    """Load and convert an image to a torch.Tensor."""
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def preprocess_image(image: torch.Tensor, resize: int, edge_divisible_by: Optional[int] = 32) -> dict:
    """
    Preprocess the image by resizing its shortest side to a target size and ensuring divisibility.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        resize (int): The target size for the shortest side.
        edge_divisible_by (int, optional): Ensures output dimensions are divisible by this value. Default is None.

    Returns:
        dict: Processed image and metadata.
    """
    h, w = image.shape[-2:]
    scale = resize / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Adjust dimensions to be divisible by `edge_divisible_by`
    if edge_divisible_by is not None:
        new_h = (new_h // edge_divisible_by) * edge_divisible_by
        new_w = (new_w // edge_divisible_by) * edge_divisible_by

    # Resize the image
    resized_image = kornia.geometry.transform.resize(image.unsqueeze(0), (new_h, new_w)).squeeze(0)

    scale_tensor = torch.tensor([new_w / w, new_h / h], dtype=torch.float32)
    transform_matrix = np.diag([scale_tensor[0].item(), scale_tensor[1].item(), 1])
    return {
        "image": resized_image,
        "scales": scale_tensor,
        "original_image_size": np.array([w, h]),
        "image_size": np.array([new_w, new_h]),
        "transform": transform_matrix,
    }


class HPatchesDataset(Dataset):
    ignored_scenes = (
        # "i_contruction",
        # "i_crownnight",
        # "i_dc",
        # "i_pencils",
        # "i_whitebuilding",
        # "v_artisans",
        # "v_astronautis",
        # "v_talent",
    )

    def __init__(self,
                 data_dir: str,
                 resize: int = 480,
                 subset: Optional[str] = None,
                 ignore_large_images: bool = True,
                 grayscale: bool = True):
        """
        Initialize HPatches dataset.

        Args:
            data_dir (str): Directory containing the HPatches dataset.
            resize (int): Image resizing dimension (default: 480).
            subset (str, optional): Use subset of data ('i' for illumination, 'v' for viewpoint). Default is None (use all).
            ignore_large_images (bool): Ignore specific large images listed in ignored_scenes. Default is True.
            grayscale (bool): Load images in grayscale if True, otherwise load in color. Default is False.
        """
        super().__init__()
        logger.info(f"Creating dataset {self.__class__.__name__}")
        self.data_dir = Path(data_dir + "/hpatches-sequences-release")
        self.resize = resize
        self.subset = subset
        self.ignore_large_images = ignore_large_images
        self.grayscale = grayscale

        if not self.data_dir.exists():
            logger.error("Dataset directory not found. Please check the path.")
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        self.sequences = sorted([x.name for x in self.data_dir.iterdir() if x.is_dir()])
        if not self.sequences:
            raise ValueError("No image sequences found in the dataset directory!")

        self.items = self._init_items()

    def _init_items(self):
        """Initialize dataset items based on sequences."""
        items = []  # (sequence_name, query_index, is_illumination)
        for seq in self.sequences:
            if self.ignore_large_images and seq in self.ignored_scenes:
                continue
            if self.subset is not None and self.subset != seq[0]:
                continue
            for i in range(2, 7):  # Indices 2-6 are used for matching
                items.append((seq, i, seq[0] == "i"))  # 'i' indicates illumination scenes
        return items

    def _read_image(self, seq: str, idx: int) -> dict:
        """Read and preprocess an image."""
        img_path = self.data_dir / seq / f"{idx}.ppm"
        img = load_image(img_path, grayscale=self.grayscale)
        return preprocess_image(img, resize=self.resize)

    def __len__(self):
        """Return the total number of items."""
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample from the dataset."""
        seq, q_idx, is_illu = self.items[idx]

        # Load and preprocess images
        data0 = self._read_image(seq, 1)
        data1 = self._read_image(seq, q_idx)

        # Read and transform homography
        H_path = self.data_dir / seq / f"H_1_{q_idx}"
        H = read_homography(H_path)
        H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])

        return {
            "H_0to1": H.astype(np.float32),
            "scene": seq,
            "idx": idx,
            "is_illu": is_illu,
            "name": f"{seq}/{q_idx}.ppm",
            "image0": data0["image"],
            "image1": data1["image"],
            # "scale0": data0["scales"],
            # "scale1": data1["scales"],
            "scale0": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "scale1": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "image0_size": data0["image_size"],
            "image1_size": data1["image_size"],
        }
