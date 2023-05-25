import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import io

def figure_to_array(fig:plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    try:
        ret = np.array(fig.canvas.renderer._renderer)
    except:
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="raw")
        io_buf.seek(0)
        ret = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            (int(fig.bbox.bounds[-1]), int(fig.bbox.bounds[-2]), -1)
        )
    return ret


def draw_heatmap_with_colorbar(
    scores:np.ndarray, 
    image:np.ndarray=None, 
    alpha=0.2, 
    figsize=(15,15), 
    cmap='jet',
    vrange:tuple=None,
    savepath=None,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    vmin, vmax = vrange if vrange is not None else (scores.min(), scores.max())
    im = ax.imshow(scores, cmap=cmap, vmin=vmin, vmax=vmax)
    if image is not None:
        ax.imshow(image)
        ax.imshow(scores, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("anomaly score map")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr[:,:,:3]
    

def draw_heatmap_with_colorbar_with_image(
    scores:np.ndarray, 
    image:np.ndarray, 
    alpha=0.2, 
    figsize=(15,15), 
    cmap='jet',
    vrange:tuple=None,
    savepath=None,
):
    fig = plt.figure(figsize=figsize)
    ax_img, ax = fig.subplots(2,1)
    ax_img.imshow(image)
    vmin, vmax = vrange if vrange is not None else (scores.min(), scores.max())
    im = ax.imshow(scores, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(image)
    ax.imshow(scores, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("anomaly score map")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr[:,:,:3]
    

def visualize_tsne(data, label=None, savename=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(verbose=1)
    print("Start TSNE")
    tsne_data = tsne.fit_transform(data)
    print("TSNE Done")
    x_new = tsne_data[:, 0]
    y_new = tsne_data[:, 1]
    
    fig, ax = plt.subplots()
    if label:
        label_v = np.unique(label)
        for l in label_v:
            ax.scatter(
                x_new[label==l], y_new[label==l],
                alpha=0.2, label=f"{l}"
            )
    else:
        ax.scatter(x_new, y_new, alpha=0.2)

    fig.tight_layout()
    if savename is not None: fig.savefig(savename)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr[:,:,:3]