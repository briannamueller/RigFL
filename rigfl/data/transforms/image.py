"""Image conversion shared by the registered image datasets."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage


def image_tensor(values, *, mean=None, std=None, image_mode=None) -> torch.Tensor:
    tensors = []
    for value in values:
        if image_mode is not None:
            mode = "RGB" if image_mode == "rgb" else "L"
            if not isinstance(value, PILImage.Image):
                value = PILImage.fromarray(np.asarray(value))
            value = value.convert(mode)
        array = np.asarray(value)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError(
                f"image input must have 2 or 3 dimensions, got {array.shape}"
            )
        tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float()
        if np.issubdtype(array.dtype, np.integer):
            tensor = tensor / float(np.iinfo(array.dtype).max)
        tensors.append(tensor)
    try:
        output = torch.stack(tensors)
    except RuntimeError as exc:
        shapes = sorted({tuple(tensor.shape) for tensor in tensors})
        raise ValueError(
            f"images have inconsistent shapes {shapes}; register a transform that "
            "resizes this dataset"
        ) from exc
    if mean is not None:
        if output.shape[1] != len(mean):
            raise ValueError(
                f"normalization defines {len(mean)} channels but inputs have "
                f"{output.shape[1]}"
            )
        mean_tensor = torch.tensor(mean, dtype=output.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=output.dtype).view(1, -1, 1, 1)
        output = (output - mean_tensor) / std_tensor
    return output
