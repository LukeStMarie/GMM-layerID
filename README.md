# GMM-layerID

This code allows for the identification of flakes of 2D materials and the differentiation of different layer thicknesses of the flakes.

## Dependencies

This code requires openCV, NumPy, SciPy, Matplotlib, and scikit-learn.

## Usage

To use this code, two sets of images are required. One set is the training set, and should contain images of 2D material flakes. The flakes should all be the same material on the same substrate, and should have similar lighting conditions. The images should be cropped so that they contain only flakes of the desired material and the bare substrate, and contain as little debris, residue, bulk material, and other "junk" as possible. The user should generally know what layer thicknesses are present in these images, although they don't need to know the exact thickness of every part of every flake.

The other set is the test set. It a set of images of flakes of the same material on the same substrate, with similar lighting conditions as the training set. These images do not need to be cropped, and can contain "junk".

To use the code, run the `training` function, which takes the following arguments: `img_dir`, the path of the folder containing the set of training images; `n_clusters`, which is the number of different layer thicknesses present in the training image set; and `out_file` the filename of where the output of the training should be saved.

In the example below, the training images in 'CropImageSetB' contain flakes with monolayer, bilayer, 3-layer, and 4-layer graphene, as well as bare substrate, so there are 5 layer thicknesses present in the images.

    args1 = {'img_dir': "CropImageSetB",
    'n_clusters': 5,
    'out_file': 'CropImageSetB/master_catalog_5.npz'}
    training(**args1)

Run training, and the catalog file will be created for this image set. Now, you can use this catalog file to run the `testing` function on an image containing flakes of unknown thickness.  This functions takes three arguments: `img_dir`, the path of he folders containing the images to test; `n_clusters`, which should be the same as used in training; and `master_cat_file`, the path of the catalog file produced by `training`.

    args2 = {'img_dir': 'ImageSetB',
            'n_clusters': 5,
            'master_cat_file': 'CropImageSetB/master_catalog_5.npz'}
    testing(**args2)

You probably want to first run `testing` on the training image set, to make sure it worked how you expected, before running it on your test image set.


