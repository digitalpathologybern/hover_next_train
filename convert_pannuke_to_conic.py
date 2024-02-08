import os
import argparse
import numpy as np
from scipy.ndimage import label

# quick and dirty script to convert pannuke masks to lizard/conic format
# define path_to_pannuke to point to the folder containing the pannuke masks
# the folder should contain 3 subfolders: fold1, fold2, fold3 with the respective masks.npy files
# the script will create a labels.npy file in each of the 3 subfolders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="pannuke/masks",
        help="specify the path to the pannuke masks folder",
    )
    path_to_pannuke = parser.parse_args().path
    for f in ["fold1", "fold2", "fold3"]:
        y = np.load(f"{path_to_pannuke}/{f}/masks.npy")

        y_inst = np.stack(
            [label(y_)[0] for y_ in np.sum(y[..., :-1], axis=-1).astype(int)]
        )
        y_cls = np.max((y[..., :-1] > 0) * np.arange(1, 6), axis=-1)
        y_lab = np.stack([y_inst, y_cls], axis=-1)
        np.save(
            os.path.join(
                path_to_pannuke,
                f,
                "labels.npy",
            ),
            y_lab,
        )
    print("done")
