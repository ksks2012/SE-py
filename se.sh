#!/bin/bash

# ========================================================================
# Copyright (c) 2015-2016 by Chun-Wei Tsai.  All rights reserved.
#
# Permission to use, copy, modify, and distribute this software and
# its documentation is governed by the terms that can be found in the
# COPYRIGHT file.
# ========================================================================

# This is the script for running the program.  The parameters to the
# program are as follows, in the order to be specified:
#
# [1] number of runs
# [2] number of iterations
# [3] number of searchers
# [4] number of regions
# [5] number of samples
# [6] number of players
# [7] evaluation enable
# [8] number of function
#
#
# In brief, the command is:
#
clear
#gdb --args ./se.out 1 1000 1000 4 4 2 3 1
/usr/bin/python3 "/data/SE py/SE.py" 1 1000 4 4 2 3 1 23
