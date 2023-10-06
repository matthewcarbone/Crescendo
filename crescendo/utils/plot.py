from typing import Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def heatmap_of_lines(
    data: np.ndarray,
    title="",
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    num_yticks=5,
    xlabel="indices",
    ylabel="values",
    norm_data: Union[Literal["minmax"], None] = None,
    norm_heatmap: Union[matplotlib.colors.Normalize, None] = LogNorm,
):
    """
    Generate a heatmap from multiple lines of data.

    Returns:
    - matplotlib.figure.Figure: Generated heatmap figure.

    Example:
    ```python
        import numpy as np
        from matplotlib.colors import LogNorm

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        data_2d = np.array([signal + np.random.normal(size=100) for _ in range(100)])
        heatmap_of_lines(data=data_2d)

    ```
    """

    # initialize heatmap to zeros
    width = data.shape[1]
    height = width if height == "same" else height
    heatmap = np.zeros((width, height))

    if norm_data is not None:
        if norm_data != "minmax":
            raise NotImplementedError(f"norm_data={norm_data} not implemented")
        data = (data - data.min()) / (data.max() - data.min())

    # compute heatmap values
    temp_data = (data - data.min()) / (data.max() - data.min())
    x_idx = np.floor(temp_data * width).astype(int) - 1
    y_idx = np.array([np.arange(width)] * temp_data.shape[0]) - 1
    for i, j in zip(x_idx, y_idx):
        heatmap[i, j] += 1

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin="lower", cmap=cmap, norm=norm_heatmap())
    plt.colorbar()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    yticks_values = np.linspace(data.min(), data.max(), num=num_yticks)
    yticks_indices = np.floor(
        (yticks_values - data.min()) / (data.max() - data.min()) * width
    ).astype(int)
    plt.yticks(yticks_indices, [f"{val:.2f}" for val in yticks_values])

    return fig