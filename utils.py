#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li, Inc. All Rights Reserved
#
########################################################################

"""
File: utils.py
Author: leowan
Date: 2018/11/18 13:51:55
"""

def read_zipped_column_indexed_data_from_svmlight_file(fname):
    """
        Load data in column-indexed format
        Return:
            [[(col11, val11), (co112, val12), ..], ..], [label0, label1, ..]
    """
    X_arr = []
    y_arr = []
    with open(fname, 'r') as fd:
        for line in fd:
            line = line.rstrip()
            elem = line.split(' ')
            if len(elem) < 2:
                continue
            y_arr.append(int(elem[0]))
            X_raw = elem[1:]
            X_zipped = []
            for indval in X_raw:
                ind, val = indval.split(':')
                X_zipped.append((int(ind), float(val)))
            X_arr.append(list(zip(*X_zipped)))
    #X_ind_arr, X_val_arr = list(zip(*X_arr))
    return X_arr, y_arr

def convert_to_column_indexed_data(X_zipped_arr, y_arr):
    """
        Convert zipped data such as
            [[(col11, val11), (co112, val12), ..], ..], [label0, label1, ..]
        to
            [[col11, col12,..], [col21, col22], ..],
            [[val11, val12,..], [val21, val22], ..],
            [label0, label1, ..]
    """
    X_ind_arr, X_val_arr = list(zip(*X_zipped_arr))
    return X_ind_arr, X_val_arr, y_arr

def convert_to_fully_column_indexed_data(X_colind_arr, X_colval_arr, y_arr, feature_num):
    """
        Fill-up with zeros
        Convert data (e.g. feature_num = 4) such as:
            [[2, 3], ..],
            [[1, 1], ..],
            [1, ..]
        to
            [[0, 1, 2, 3], ..],
            [[0, 0, 1, 1], ..],
            [1, ..]
    """
    X_colind_full_arr = []
    X_colval_full_arr = []
    assert len(X_colind_arr) == len(X_colval_arr) and len(X_colind_arr) == len(y_arr), 'Mismatch: len'
    for i in range(len(y_arr)):
        full_ind = list(range(feature_num))
        full_val = [0] * feature_num
        for offset, colind in enumerate(X_colind_arr[i]):
            full_val[colind] = X_colval_arr[i][offset]
        X_colind_full_arr.append(full_ind)
        X_colval_full_arr.append(full_val)
    return X_colind_full_arr, X_colval_full_arr, y_arr