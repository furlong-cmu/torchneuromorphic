#!/bin/python
#-----------------------------------------------------------------------------
# File Name : create_dvsgestures.py
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 13:20:05 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchneuromorphic.dvs_gestures.create_hdf5 import *

if __name__ == "__main__":
    out = create_events_hdf5('/Users/ZP/Downloads/DVSdataset/DVSGesture', '/Users/ZP/Downloads/DVSbuild/rotfix/dvs_gestures_build19.hdf5')




