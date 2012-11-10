result_stats
============

Utility that, given two score files from Positive and Negative examples, 
computes different statistics from the classifier. 
Project proposed in Unit 2. Max. points = 2.5.

Statistics
----------

* [**ROC Curve**](http://en.wikipedia.org/wiki/Receiver_operating_characteristic), shows how the FPR and 1 - FNR change using different threshold values.
* **Area under ROC Curve**, single value indicating the quality of the classifier. The higher area, the better.
* **FPR at a given FNR**, computes the threshild that gives the lowest FPR that satisfies that the FNR is lower or equal to the given one. The lower value, the better.
* **FNR at a given FPR**, computes the threshold that gives the lowest FNR that satisfies that the FPR is lower or equal to the given one. The lower value, the better.
* **FPR = FNR**, computes the threshold that gives the smallest difference between FPR and FNR. In case of many minimums, obtains the threshold that minimizes FPR + FNR.
* [*D-Prime*](http://en.wikipedia.org/wiki/D'), computes the separation between the positive and negative scores assuming a Normal distribution. The higher value, the better.

Notes: 
* **FPR** means [*False Positive Rate*](http://en.wikipedia.org/wiki/False_positive#Type_I_error).
* **FNR** means [*False Negative Rate*](http://en.wikipedia.org/wiki/False_positive#Type_II_error).


Requirements
------------

* Python v2.7, this software was tested using Python 2.7. It may work with prior and posterior versions, but it is not guaranteed.
* matplotlib, used to plot the ROC curve. http://matplotlib.org


Help
----
    Usage:
      ./result_stats.py --pos pfile --neg nfile [--fpr FPR] [--fnr FNR]
    
    Options:
      --help (-h)         Shows this text
      --pos (-p) pfile    Scores file of the possitive examples
      --neg (-n) nfile    Scores file of the negative examples
      --fpr FPR           Desired FPR (Range: 0..1. Default: 0.05)
      --fnr FNR           Desired FNR (Range: 0..1. Default: 0.05)
