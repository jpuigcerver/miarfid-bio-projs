#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import math
import matplotlib.pyplot as pyplot

def ParseArguments():
    """Parse the command line arguments."""
    pos_file = None
    neg_file = None
    fpr, fnr = 0.05, 0.05
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ('--pos', '-p'):
            pos_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] in ('--neg', '-n'):
            neg_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--fpr':
            fpr = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--fnr':
            fnr = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] in ('--help', '-h'):
            ShowHelp()
            sys.exit(0)
        else:
            ShowHelp()
            sys.exit(1)
    if fpr < 0 or fnr < 0 or fpr > 1 or fnr > 1:
        ShowHelp()
        sys.exit(1)
    if pos_file is None or neg_file is None:
        ShowHelp()
        sys.exit(1)
    return (pos_file, neg_file, fpr, fnr)

def ShowHelp():
    print('Usage:')
    print('  %s --pos pfile --neg nfile [--fpr FPR] [--fnr FNR]' % sys.argv[0])
    print('\nOptions:')
    print('  --help (-h)         Shows this text')
    print('  --pos (-p) pfile    Scores file of the possitive examples')
    print('  --neg (-n) nfile    Scores file of the negative examples')
    print('  --fpr FPR           Desired FPR (Range: 0..1. Default: 0.05)')
    print('  --fnr FNR           Desired FNR (Range: 0..1. Default: 0.05)')

def LoadScores(pos_filename, neg_filename):
    """Load the data from the positive and negative examples and merges it.
    In case that the input files have multiple columns, the last column will
    be considered as the score of the example.
    The data points are sorted in order of increasing score.
    """
    scores = []
    # Read pos file
    f = open(pos_filename, 'r')
    for line in f:
        line = line.split()
        scores.append((float(line[-1]), '+'))
    f.close()
    # Read neg file
    f = open(neg_filename, 'r')
    for line in f:
        line = line.split()
        scores.append((float(line[-1]), '-'))
    f.close()
    scores.sort()
    return scores

def GetInterestingThresholds(scores):
    """Given the sorted list of scores, get the different thresholds 
    worth to try (removes repeated scores). An extra threshold is
    in the beggining to make sure that there is an scenario with
    FPR = 1.0, FNR = 0.0.
    """
    thresholds = [scores[0][0] - 1.0]
    for (sc,c) in scores:
        if len(thresholds) == 0 or sc != thresholds[-1]:
            thresholds.append(sc)
    return thresholds

def GetStats(scores, th):
    """Given the list of scores and a threshold, compute the
    number of TP, FP, TN, FN and number of positive and negative examples.
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for (sc,c) in scores:
        if sc > th:
            if c == '+':
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if c == '+':
                fn = fn + 1
            else:
                tn = tn + 1
    p = tp + fn
    n = tn + fp
    return tp, fp, tn, fn, p, n
        
def GetStatsAllThresholds(scores):
    """Compute the FPR and FNR for all the interesting thresholds."""
    thresholds = GetInterestingThresholds(scores)
    stats = []
    for th in thresholds:
        tp, fp, tn, fn, p, n = GetStats(scores, th)
        fpr = fp / float(n)
        fnr = fn / float(p)
        stats.append((th, fpr, fnr))
    return stats

def PlotRocCurve(stats):
    """Given the list of FPR and FNR for each threshold, plots
    the ROC curve.
    """
    pyplot.axis([0, 1, 0, 1])
    X = [x for th, x, y in stats]
    Y = [1 - y for th, x, y in stats]
    pyplot.plot(X, Y)
    pyplot.xlabel('FPR')
    pyplot.ylabel('1 - FNR')
    pyplot.show()

def GetFPAtGivenFN(stats, given_fn):
    """Returns the threshold and the smallest FPR which holds that
    FNR <= given_fn.
    """
    for i in range(len(stats)-1, -1, -1):
        (th, fp, fn) = stats[i]
        if fn <= given_fn:
            return th, fp
    return (None, None)

def GetFNAtGivenFP(stats, given_fp):
    """Returns the threshold and the smallest FNR which holds that
    FPR <= given_fp.
    """
    for i in range(0, len(stats)):
        (th, fp, fn) = stats[i]
        if fp <= given_fp:
            return th, fn
    return None, None

def GetMinDiffFpFn(stats):
    """Returns the threshold, FPR and FNR that minimizes the distance
    between FPR and FNR. If there are multiple thresholds that assert that
    condition, returns the setting which minimizes FPR + FNR.
    """
    d, fp, fn, th = 1, 1, 0, 0
    for (thi, fpi, fni) in stats:
        nd = math.fabs(fpi-fni)
        if nd < d:
            d, fp, fn, th = nd, fpi, fni, thi
        elif nd == d and fpi + fni < fp + fn:
            d, fp, fn, th = nd, fpi, fni, thi
    return th, fp, fn

def GetRocCurveArea(stats):
    """Computes the area under the ROC curve."""
    a, fpa, fna = 0.0, stats[0][1], 1 - stats[0][2]
    for i in range(1, len(stats)):
        th, fp, fn = stats[i]
        fn = 1 - fn
        a = a + (fpa - fp) * fn
        fpa, fna = fp, fn
    return a

def GetDScore(scores):
    """Compute the D-Prime score."""
    # Compute mean and variance of each class
    M = [0.0, 0.0]
    Q = [0.0, 0.0]
    N = [0, 0]
    for (sc, c) in scores:
        if c == '+': idx = 0
        else: idx = 1
        N[idx] = N[idx] + 1
        d = sc - M[idx]
        M[idx] = M[idx] + d / N[idx]
        Q[idx] = Q[idx] + (N[idx] - 1) * d * d / N[idx]
    V = [Q[0] / (N[0] - 1), Q[1] / (N[1] - 1)]
    return (M[0] - M[1]) / math.sqrt(V[0] + V[1])

def main():
    pos_filename, neg_filename, fpr, fnr = ParseArguments()
    scores = LoadScores(pos_filename, neg_filename)
    stats = GetStatsAllThresholds(scores)
    th_fpr, fnr_fpr = GetFNAtGivenFP(stats, fpr)
    th_fnr, fpr_fnr = GetFPAtGivenFN(stats, fnr)
    th_diff, fpr_diff, fnr_diff = GetMinDiffFpFn(stats)
    print('FNR at a given FPR:')
    print('===================')
    print('  fpr = %f' % fpr)
    print('  fnr = %f' % fnr_fpr)
    print('  thr = %f' % th_fpr)
    print('FPR at a given FNR:')
    print('===================')
    print('  fpr = %f' % fnr_fpr)
    print('  fnr = %f' % fnr)
    print('  thr = %f' % th_fnr)
    print('FPR = FNR:')
    print('==========')
    print('  fpr = %f' % fpr_diff)
    print('  fnr = %f' % fnr_diff)
    print('  thr = %f' % th_diff)
    print('D-Prime score:')
    print('==============')
    print('  d = %f' % GetDScore(scores))
    print('ROC curve area:')
    print('===============')
    print('  a = %f' % GetRocCurveArea(stats))
    PlotRocCurve(stats)

if __name__ == '__main__':
    main()
