schneiderman_kanade
===================

Face detector based on the Schneiderman & Kanade technique for objects
detection. Includes a set of tools to train and test faces vs. non-faces datasets
and a face detector that can be used with many image formats, thanks to Magick++.

Tools
-----
* **sk-train** trains a face detector model based on a training and a validation set of images containing faces and non-faces.
* **sk-test** test a trained model using a test set.
* **sk-detect** use a trained model to detect faces in a regular image.
* **normalize-img** normalizes an image (mean=0, std=1) and prints the result. Useful to create datasets for sk-train and sk-test.

Scripts
-------
 * **explore-detector-parameters.sh** trains many face detectors with different configuration parameters. Useful to choose the best configuration.
 * **generate-dataset-from-dirs.sh** generates a dataset from two directories containing face images and non-face images.
 * **generate-dataset-from-files.sh** basically merges and shuffles to files containing normalized faces and non-faces.

Requirements
------------
* C++ compiler (C++11 support needed). Tested using: g++-4.7 and g++-4.8
* libblas, http://www.netlib.org/blas/. Tested using: libblas-1.2
* libgflags, http://code.google.com/p/gflags/. Tested using: libgflags-2.0
* libglog, http://code.google.com/p/google-glog/. Tested using: glog-0.3.2
* libprotobuf, http://code.google.com/p/protobuf/. Tested using: protobuf-2.4.1
* libmagick++, http://www.imagemagick.org/Magick++/. Tested using: libmagick++-6.7.7.10-2

Help
----
 * **sk-train**

    Usage:
      ./sk-train -train training_data -valid validation_data \
        -img_w img_w -img_h img_h -reg_w subregion_w -reg_h subregion_h

    Interesting flags:
      -d (Reduce regions dimensionality to d) type: uint64 default: 10
      -img_h (Training images height) type: uint64 default: 0
      -img_w (Training images width) type: uint64 default: 0
      -k (Number of quantized patterns) type: uint64 default: 10
      -kclustering_threads (Number of threads for K-Clustering algorithm)
        type: uint64 default: 8
      -mfile (File where the trained model will be written) type: string
        default: ""
      -reg_h (Subregion height) type: uint64 default: 5
      -reg_w (Subregion width) type: uint64 default: 5
      -seed (Seed for the random engine) type: uint64 default: 0
      -stp_x (Step size in x direction) type: uint64 default: 5
      -stp_y (Step size in y direction) type: uint64 default: 5
      -train (File containing the training set) type: string default: ""
      -valid (File containing the validation set) type: string default: ""
      -optimize (Criterion for threshold. Values: best_acc, fixed_fpr,
        fixed_fnr.) type: string default: "best_acc"
      -fnr (Desired FNR. Only used when optimize = fixed_fnr) type: double
        default: 0.3
      -fnr (Desired FPR. Only used when optimize = fixed_fpr) type: double
        default: 0.3

 * **sk-test**

    Usage:
      ./sk-test -test test_data -mfile trained_model

    Interesting flags:
      -mfile (File containing the trained model.) type: string default: ""
      -seed (Seed for the random engine) type: uint64 default: 0
      -test (File containing the test set) type: string default: ""

 * **sk-detect**

        Usage:
          ./sk-detect -m trained_model -i Lenna.png -o Lenna_faces.png

        Interesting flags:
          -display (Display output image) type: bool default: false
          -i (Input image. If '-', standard input.) type: string default: "-"
          -m (Schneiderman & Kanade model file) type: string default: ""
          -max_scale (Maximum scaling factor) type: double default: 1
          -min_scale (Minimum scaling factor (may be superseded)) type: double
            default: 0
          -o (Output image) type: string default: ""
          -scales (Number of times image is scaled) type: uint64 default: 10
          -step_x (Window step in x direction) type: uint64 default: 1
          -step_y (Window step in y direction) type: uint64 default: 1

All the previous tools support the flags introduced by glog and gflags, the
most interesting one is probably `-logtostderr` which will show a detailed
log of the execution of the given tool. Simply use `-help` to show all the flags
supported by any given tool.