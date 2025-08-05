from PIL import Image
import numpy as np
import cv2
from typing import Union
from scipy.spatial import ConvexHull
from skimage.draw import polygon

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}
ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}
LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
    'shoes': ['Right-arm', 'Left-arm', 'Face', 'Left-leg', 'Right-leg'],
    'bag': ['Face'],
}
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes', 'Left-shoe', 'Right-shoe'],
        'LIP': ['Upper-clothes', 'Coat', 'Left-shoe', 'Right-shoe']
    },
    'overall': {
        'ATR': ['Left-shoe', 'Right-shoe'],
        'LIP': ['Left-shoe', 'Right-shoe']
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    },
    'shoes': {
        'ATR': [],
        'LIP': []
    },
    'bag': {
        'ATR': [],
        'LIP': []
    }
}
PUBLIC_ACCESSORY_PARTS = ['Hat', 'Glove', 'Sunglasses', 'Scarf', 'Socks']  # 'Left-shoe', 'Right-shoe', 'Bag'
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',],
    'shoes': ['Left-shoe', 'Right-shoe'],
    'bag': ['Bag'],
}
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms'],
    'shoes': ['right foot', 'left foot'],
    'bag': [],
}


def random_convex_mask(height, width, min_area_ratio=0.25, jitter=10, extra_points=20):
    """
    在图像中随机位置生成一个凸多边形掩码，其包含一个面积不少于总面积1/4的扰动矩形。

    参数:
        height (int): 图像高度
        width (int): 图像宽度
        min_area_ratio (float): 最小面积比例
        jitter (int): 扰动范围（用于扩展凸多边形）
        extra_points (int): 随机扰动点数量

    返回:
        np.ndarray: 掩码数组 (0 背景, 1 前景)
    """
    rng = np.random.default_rng()
    total_area = height * width
    min_area = total_area * min_area_ratio

    # 随机生成一个矩形尺寸，满足面积要求
    aspect_ratio = rng.uniform(0.5, 2.0)  # 宽高比 [0.5, 2]
    rect_h = int(np.sqrt(min_area / aspect_ratio))
    rect_w = int(rect_h * aspect_ratio)

    # 随机偏移（使矩形尽量留在图像内）
    max_x0 = width - rect_w - 1
    max_y0 = height - rect_h - 1
    x0 = rng.integers(0, max(1, max_x0))
    y0 = rng.integers(0, max(1, max_y0))
    x1 = x0 + rect_w
    y1 = y0 + rect_h

    # 矩形四角 + 扰动点
    base_points = np.array([
        [x0, y0], [x1, y0], [x1, y1], [x0, y1]
    ])
    jitter_points = rng.integers(
        low=[max(0, x0 - jitter), max(0, y0 - jitter)],
        high=[min(width, x1 + jitter), min(height, y1 + jitter)],
        size=(extra_points, 2)
    )
    all_points = np.vstack([base_points, jitter_points])

    # 计算凸包并生成掩码
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]

    mask = np.zeros((height, width), dtype=np.uint8)
    rr, cc = polygon(hull_points[:, 1], hull_points[:, 0], shape=mask.shape)
    mask[rr, cc] = 1

    return mask

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, 
                 mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def hull_mask(mask_area: np.ndarray):
    if len(mask_area.shape) > 2:
        mask_area = mask_area[:,:,0]
    
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask

def create_square_mask(mask_area, expand_ratio=0.1):
    """
    将非零区域的最小外接矩形填充为1，并随机扩大边界
    
    Args:
        mask_area (np.ndarray): 原始 mask
        expand_ratio (float, optional): 边界随机扩大的最大比例。默认为0.1（10%）
    
    Returns:
        np.ndarray: 原始大小的 mask，矩形区域为1
    """
    # 找到非零区域
    rows = np.any(mask_area, axis=1)
    cols = np.any(mask_area, axis=0)
    
    # 获取非零区域边界
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 计算高度和宽度
    height = rmax - rmin + 1
    width = cmax - cmin + 1
    
    # 随机扩大边界
    expand_height = int(height * expand_ratio)
    expand_width = int(width * expand_ratio)
    
    # 计算新的边界，确保不超过图像尺寸
    new_rmin = max(0, rmin - np.random.randint(0, expand_height + 1))
    new_rmax = min(mask_area.shape[0] - 1, rmax + np.random.randint(0, expand_height + 1))
    new_cmin = max(0, cmin - np.random.randint(0, expand_width + 1))
    new_cmax = min(mask_area.shape[1] - 1, cmax + np.random.randint(0, expand_width + 1))
    
    # 创建新的 mask，并将矩形区域填充为1
    new_mask_area = np.zeros_like(mask_area)
    new_mask_area[new_rmin:new_rmax+1, new_cmin:new_cmax+1] = 1
    
    return new_mask_area

def create_bounding_box_mask(
    mask_area: np.ndarray,
    strong_protect_area: np.ndarray,
    dilate_kernel: np.ndarray,
    horizon_expand: bool=False
) -> np.ndarray:
    """创建包含原始mask区域的外接矩形Mask，并排除保护区域

    Args:
        mask_area (np.ndarray): 原始mask区域
        strong_protect_area (np.ndarray): 需要排除的强保护区域
        dilate_kernel (np.ndarray): 用于膨胀操作的kernel
        horizon_expand (bool): 是否启用随机水平扩展

    Returns:
        np.ndarray: 处理后的矩形mask
    """
    # 找到mask区域的非零点坐标
    coords = cv2.findNonZero(mask_area)
    if coords is not None:
        # 获取外接矩形的坐标
        x, y, w, h = cv2.boundingRect(coords)
        
        # 随机水平扩展
        if horizon_expand:
            img_width = mask_area.shape[1]
            # 左右两边独立随机扩展 0.0～0.1倍的宽度
            left_expand = int(np.random.uniform(0.0, 0.2) * w)
            right_expand = int(np.random.uniform(0.0, 0.2) * w)
            
            # 计算扩展后的坐标，确保不超出图像边界
            x_expanded = max(0, x - left_expand)
            w_expanded = min(img_width - x_expanded, w + left_expand + right_expand)
            
            x, w = x_expanded, w_expanded
        
        # 创建外接矩形mask
        rect_mask = np.zeros_like(mask_area)
        rect_mask[y:y+h, x:x+w] = 1
        # 排除强保护区域
        rect_mask = rect_mask & (~strong_protect_area)
        # 最后的膨胀操作
        rect_mask = cv2.dilate(rect_mask, dilate_kernel, iterations=1)
        return rect_mask
    return mask_area

def cloth_agnostic_mask(
    densepose_mask: np.ndarray,
    schp_lip_mask: np.ndarray,
    schp_atr_mask: np.ndarray,
    part: str='overall',
    square_cloth_mask: bool=False,
    **kwargs
) -> Image.Image:
    if part == 'full' or part == 'dresses':
        part = 'overall'
    assert part in ['upper', 'lower', 'overall', 'inner', 'outer', 'shoes', 'bag'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer', 'shoes', 'bag'], but got {part}"
    w, h = densepose_mask.shape[:2]
    
    dilate_kernel = max(w, h) // 500
    dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
    dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    
    kernal_size = max(w, h) // 50
    kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
    
    # Strong Protect Area (Hands, Face, Accessory, Feet)
    hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
    hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
    hands_protect_area = hands_protect_area & \
        (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
            part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
    face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)

    strong_protect_area = hands_protect_area | face_protect_area 

    # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
    body_protect_area = part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
    hair_protect_area = part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | \
        part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING)
    cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
        part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
    accessory_protect_area = part_mask_of(PUBLIC_ACCESSORY_PARTS, schp_lip_mask, LIP_MAPPING) | \
        part_mask_of(PUBLIC_ACCESSORY_PARTS, schp_atr_mask, ATR_MAPPING) 
    weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | strong_protect_area | accessory_protect_area
    
    # Mask Area
    strong_mask_area = part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
        part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
    strong_mask_area = cv2.dilate(strong_mask_area, dilate_kernel // 2 +1, iterations=1)  # ADD: max pooling
    background_area = part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING)
    mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
    mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
    mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    mask_area = (mask_dense_area | strong_mask_area & (~weak_protect_area) & (~background_area))
    
    # 边缘平滑（去除毛刺）
    mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
    mask_area[mask_area < 100] = 0
    mask_area[mask_area >= 100] = 1
    
    # 确保只有2个以下连通域
    # num_labels, labels = cv2.connectedComponents(mask_area.astype(np.uint8))
    # if num_labels > 3:  # 背景(0)和一个前景区域
    #     label_counts = np.bincount(labels.flatten()) # 找到最大连通区域的标签
    #     label_counts[0] = 0  # 排除背景
    #     largest_label = np.argmax(label_counts)
    #     mask_area = (labels == largest_label).astype(np.uint8)  # 创建只包含最大连通区域的掩码，并填充内部空洞

    
    # 凸包扩张
    mask_area = hull_mask(mask_area * 255) // 255  # Convex Hull to expand the mask area
    mask_area = mask_area & (~weak_protect_area)
    mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
    mask_area[mask_area < 25] = 0
    mask_area[mask_area >= 25] = 1
    
    # 创建方形mask
    if square_cloth_mask: # v2
        weak_protect_area_ = weak_protect_area & (~mask_area) # 防止边缘过度保护
        mask_area = create_bounding_box_mask(
            mask_area, 
            weak_protect_area_, 
            dilate_kernel
        )
        mask_area = mask_area & ~face_protect_area
        
    # Pooling
    mask_area = cv2.dilate(mask_area, dilate_kernel - 2, iterations=2)
    
    return Image.fromarray(mask_area * 255)
    
def multi_ref_cloth_agnostic_mask(
    densepose_mask: np.ndarray,
    schp_lip_mask: np.ndarray,
    schp_atr_mask: np.ndarray,
    square_cloth_mask: bool=False,
    horizon_expand: bool=False,
    **kwargs
) -> Image.Image:
    w, h = densepose_mask.shape[:2]
    
    dilate_kernel = max(w, h) // 500
    dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
    dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    
    kernal_size = max(w, h) // 50
    kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
    
    # Strong Protect Area (Hands, Face, Accessory, Feet)
    # hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
    # hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
    # hands_protect_area = hands_protect_area & \
    #     (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
    #         part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
    face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)
    # strong_protect_area = hands_protect_area | face_protect_area
    strong_protect_area = face_protect_area
    
    # Mask Area
    mask_keys = ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits',
                 'Left-shoe', 'Right-shoe','Bag', 'Socks', 'Belt']
    strong_mask_area = part_mask_of(mask_keys, schp_lip_mask, LIP_MAPPING) | \
        part_mask_of(mask_keys, schp_atr_mask, ATR_MAPPING)
    strong_mask_area = cv2.dilate(strong_mask_area, dilate_kernel // 2 + 1, iterations=1)  # ADD: max pooling
    mask_dense_area = part_mask_of(MASK_DENSE_PARTS['overall'], densepose_mask, DENSE_INDEX_MAP)
    mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
    mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    mask_area = strong_mask_area | mask_dense_area & (~strong_protect_area)
    mask_area = cv2.dilate(mask_area, dilate_kernel // 2 + 1, iterations=1)  # ADD: max pooling
    
    # 边缘平滑（去除毛刺）
    mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
    mask_area[mask_area < 100] = 0
    mask_area[mask_area >= 100] = 1
    
    # 确保只有一个连通域
    num_labels, labels = cv2.connectedComponents(mask_area.astype(np.uint8))
    if num_labels > 2:  # 背景(0)和一个前景区域
        label_counts = np.bincount(labels.flatten()) # 找到最大连通区域的标签
        label_counts[0] = 0  # 排除背景
        largest_label = np.argmax(label_counts)
        mask_area = (labels == largest_label).astype(np.uint8)  # 创建只包含最大连通区域的掩码，并填充内部空洞

    # 凸包扩张
    mask_area = hull_mask(mask_area * 255) // 255  # Convex Hull to expand the mask area
    mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
    mask_area[mask_area < 25] = 0
    mask_area[mask_area >= 25] = 1
    
    # 创建方形mask
    if square_cloth_mask: # v2
        # weak_protect_area_ = weak_protect_area & (~mask_area) # 防止边缘过度保护
        mask_area = create_bounding_box_mask(
            mask_area, 
            strong_protect_area,
            dilate_kernel,
            horizon_expand
        )
        mask_area = mask_area & ~strong_protect_area
        
    # Pooling
    mask_area = cv2.dilate(mask_area, dilate_kernel - 2, iterations=2)
    
    return Image.fromarray(mask_area * 255)




