import numpy as np
import matplotlib.pyplot as plt
from dataset import *


train_or_val = 'train'
full_or_empty = 'full'


def min_max_images(segmentation_dataset, save_output):
    h_vect = []
    w_vect = []
    values = []
    sizes = np.zeros((650, 650), dtype=int)

    print("Filling the table...")
    for i in range(len(segmentation_dataset)):
        h = segmentation_dataset[i]['input'].shape[0]
        w = segmentation_dataset[i]['input'].shape[1]

        sizes[h][w] += 1

        if ((i+1) % 100 == 0):
            print(i+1, "th image")

    print("Creating the arrays...")
    for h in range(650):
        for w in range(650):
            if sizes[h][w] != 0:
                h_vect.append(h)
                w_vect.append(w)
                values.append(sizes[h][w])

    print("Minimum height is", np.min(h_vect), "pixels")
    print("Maximum height is", np.max(h_vect), "pixels")
    print("Minimum width is", np.min(w_vect), "pixels")
    print("Maximum width is", np.max(w_vect), "pixels")

    print("Plotting...")
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.013
    plt.figure("Data characterization", figsize=(5.5, 5.5))

    # Scatter
    rect_scatter = [left, bottom, width, height]
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_scatter.set_xlabel("Images widths")
    ax_scatter.set_ylabel("Images heights")
    ax_scatter.scatter(w_vect, h_vect, s=[v/10 for v in values], c=values)
    lim_inf = np.ceil(np.abs([w_vect, h_vect]).min()) - 20
    lim_sup = np.ceil(np.abs([w_vect, h_vect]).max()) + 20
    ax_scatter.set_xlim((lim_inf, lim_sup))
    ax_scatter.set_ylim((lim_inf, lim_sup))

    # Histograms
    bins = range(np.abs([w_vect, h_vect]).min(),
                 np.abs([w_vect, h_vect]).max(),
                 10)

    # Hist_x
    rect_histx = [left, bottom + height + spacing, width, 0.15]
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histx.hist(w_vect, bins=bins)
    ax_histx.set_xlim(ax_scatter.get_xlim())

    # Hist_y
    rect_histy = [left + width + spacing, bottom, 0.15, height]
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.hist(h_vect, bins=bins, orientation='horizontal')
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.savefig("./results/graphs/" + save_output)
    plt.savefig("./../report/pics/" + save_output)
    plt.show()


if __name__ == "__main__":

    root_dir = './data/'
    input_dir = train_or_val + '2017' + full_or_empty + '/'
    target_dir = train_or_val + '2017' + full_or_empty + '/'

    segmentation_dataset = SegmentationDataset(
            root_dir=root_dir,
            input_dir=input_dir,
            target_dir=target_dir,
            transform=None)

    save_output = train_or_val + '2017' + full_or_empty + '.png'
    min_max_images(segmentation_dataset, save_output)
