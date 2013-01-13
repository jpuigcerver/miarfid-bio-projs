lnorm
=====

Computes the local normalization of an input image. Image must be grayscale
or truecolor, alpha channel also supported.

Requirements
------------
* C++ compiler (C++11 support needed). Tested using: g++-4.7 and g++-4.8
* libgflags, http://code.google.com/p/gflags/. Tested using: libgflags-2.0
* libglog, http://code.google.com/p/google-glog/. Tested using: glog-0.3.2
* libmagick++, http://www.imagemagick.org/Magick++/. Tested using: libmagick++-6.7.7.10-2

Help
----
    Usage:
      ./lnorm -i Lenna.png -o Lenna_norm.png [-s step] [-w window] [-helpshort]

    Options:
      -i (Input image. Use '-' for standard input.) type: string default: "-"
      -o (Output image. Use '-' for standard output.) type: string default: "-"
      -s (Pixel step.) type: uint64 default: 1
      -w (Window size for local normalization.) type: uint64 default: 5

