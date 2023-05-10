from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from fpt.data import set_ax_locator


def draw_grid_gallery(target_imgs, cols=5):
    cols = 5
    SIZE = 80
    rows = len(target_imgs) // cols + 1
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(4 * cols, 4 * rows),
        gridspec_kw={"wspace": 0.1},
        constrained_layout=True,
        squeeze=False,
    )
    for index, img in enumerate(target_imgs):
        r, c = index // cols, index % cols
        ax = axs[r][c]
        ax.imshow(np.asarray(Image.open(img).resize((SIZE, SIZE))))
        ax.set_xlabel(img.split("/")[-2])
        ax = set_ax_locator(ax, (SIZE, SIZE))
