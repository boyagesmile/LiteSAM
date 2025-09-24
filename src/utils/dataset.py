import io
from loguru import logger

import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv

try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None


# --- DATA IO ---

def load_array_from_s3(
        path, client, cv_type,
        use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
        else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def imread_color(path, augment_fn=None, client=None):
    """
    读取并返回彩色图像（默认 BGR -> RGB）。

    - 若 `augment_fn` 为空，则直接返回 RGB 彩色图像。
    - 若 `augment_fn` 不为空，则先转换 RGB，应用增强函数 `augment_fn`，然后返回增强后的彩色图像。

    :param path: 图像文件路径
    :param augment_fn: 可选的数据增强函数（输入 RGB 图像，输出增强后 RGB 图像）
    :param client: 若读取 S3 存储，可传入 S3 客户端（默认 None）
    :return: 彩色 RGB 图像 (H, W, 3)
    """

    if str(path).startswith('s3://'):
        # 从 S3 读取
        image = load_array_from_s3(str(path), client, cv2.IMREAD_COLOR)
    else:
        # 读取本地图像
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"无法读取图像: {path}")

    # OpenCV 读取的是 BGR 格式，需转换为 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 若提供数据增强函数，则应用
    if augment_fn is not None:
        image = augment_fn(image)

    return image  # 返回 RGB 彩色图像


# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_color(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    读取 MegaDepth 数据集的 **灰度** 和 **彩色** 图像，并调整大小。

    Args:
        path (str): 图像文件路径。
        resize (int, optional): 调整图像大小，使最长边等于 `resize`（若为 None 则不调整）。
        df (int, optional): 使尺寸可被 `df` 整除（若为 None 则不做此约束）。
        padding (bool): 是否进行零填充（仅对灰度图生效）。
        augment_fn (callable, optional): 对 **彩色图** 进行增强（不会影响灰度图）。

    Returns:
        image (torch.tensor): (1, h, w) 归一化的灰度图像
        image_color (torch.tensor): (3, h, w) 归一化的 RGB 彩色图像
        mask (torch.tensor, optional): (h, w) 仅适用于灰度图
        scale (torch.tensor): [w/w_new, h/h_new] 尺度变换因子
    """

    # 读取 **灰度** 图像
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # 读取 **彩色** 图像 (H, W, 3) -> RGB 格式
    image_color = imread_color(path, augment_fn, client=MEGADEPTH_CLIENT)

    # 获取原始尺寸
    h, w = image.shape[:2]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    # 调整大小
    image = cv2.resize(image, (w_new, h_new))  # 灰度图
    image_color = cv2.resize(image_color, (w_new, h_new))  # 彩色图
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    # 处理 **padding**
    if padding:
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None
    # 归一化 & 转换格式
    image = torch.from_numpy(image).float()[None] / 255.0  # (H, W) -> (1, H, W)
    image_color = torch.from_numpy(image_color).float().permute(2, 0, 1) / 255.0  # (H, W, 3) -> (3, H, W)
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, image_color, mask, scale


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_color(path, resize=(640, 480), augment_fn=None):
    """
    读取 ScanNet 数据集的 **灰度** 和 **彩色** 图像，并调整大小。

    Args:
        path (str): 图像文件路径。
        resize (tuple): 目标尺寸 `(w, h)`，用于对齐深度图像。
        augment_fn (callable, optional): 仅对 **彩色图** 进行增强（不会影响灰度图）。

    Returns:
        image (torch.tensor): (1, h, w) 归一化的灰度图像
        image_color (torch.tensor): (3, h, w) 归一化的 RGB 彩色图像
        scale (torch.tensor): [w/w_new, h/h_new] 尺度变换因子
    """

    # 读取 **灰度** 图像
    image = imread_gray(path, augment_fn)

    # 读取 **彩色** 图像 (H, W, 3) -> RGB 格式
    image_color = imread_color(path, augment_fn)

    # 获取原始尺寸
    h, w = image.shape[:2]
    w_new, h_new = resize

    # 调整大小
    image = cv2.resize(image, (w_new, h_new))  # 灰度图
    image_color = cv2.resize(image_color, (w_new, h_new))  # 彩色图

    # 归一化 & 转换格式
    image = torch.from_numpy(image).float()[None] / 255.0  # (H, W) -> (1, H, W)
    image_color = torch.from_numpy(image_color).float().permute(2, 0, 1) / 255.0  # (H, W, 3) -> (3, H, W)

    return image, image_color


def read_scannet_depth(path):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]
