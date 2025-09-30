"""
Utility functions for extracting image and mask patches using sliding window.
"""

__all__ = ['get_image_patches', 'get_mask_patches']


def get_image_patches(image_np, patch_size, stride):
    """
    Extract patches from a 2D or 3D image using sliding window.

    Args:
        image_np (np.ndarray): Input image (2D or 3D).
        patch_size (tuple): (patch_height, patch_width).
        stride (tuple): (stride_height, stride_width).

    Returns:
        list[np.ndarray]: Extracted image patches.
    """
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    if image_np.ndim == 3:
        height, width, _ = image_np.shape
    elif image_np.ndim == 2:
        height, width = image_np.shape
    else:
        raise ValueError("image_np must be 2D or 3D.")

    patches = []
    for y in range(0, height - patch_height + 1, stride_height):
        for x in range(0, width - patch_width + 1, stride_width):
            if image_np.ndim == 3:
                patch = image_np[y:y + patch_height, x:x + patch_width, :]
            else:
                patch = image_np[y:y + patch_height, x:x + patch_width]
            patches.append(patch)

    return patches


def get_mask_patches(mask_np, patch_size, stride):
    """
    Extract patches from a 2D mask using sliding window.

    Args:
        mask_np (np.ndarray): 2D instance mask.
        patch_size (tuple): (patch_height, patch_width).
        stride (tuple): (stride_height, stride_width).

    Returns:
        list[np.ndarray]: Extracted mask patches.
    """
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    height, width = mask_np.shape
    patches = []

    for y in range(0, height - patch_height + 1, stride_height):
        for x in range(0, width - patch_width + 1, stride_width):
            patch = mask_np[y:y + patch_height, x:x + patch_width]
            patches.append(patch)

    return patches
