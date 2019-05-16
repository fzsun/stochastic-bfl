# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:37:26 2019

@author: fangzhou
"""

import os
import errno
import pandas as pd
import numpy as np
import glob
from loguru import logger

def ins_parser(ins_path):
    """
    Parse an UflLib instance file given the path to that instance

    Input: file path to standard UflLib-format data.

    Format description see
    http://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib/

    Return (num_city, num_facility, opening_cost, connection_cost).
    """
    with open(ins_path, "r") as f:
        line1 = f.readline()
        line2 = f.readline()
    num_facility, num_city, _ = list(map(int, line2.split()))
    df = pd.read_csv(ins_path, sep=" ", index_col=0, header=None, skiprows=2)
    opening_cost = df[1]
    connection_cost = df[np.arange(2, num_city + 2)].T
    return num_city, num_facility, opening_cost.to_numpy(), connection_cost.to_numpy()


def ins_path_finder(ufllib_dir):
    """
    Iteratively find all instance paths inside the UflLib directory.

    Return instance path.
    """
    if not os.path.exists(ufllib_dir):
        raise FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), ufllib_dir)
    for list_file in glob.iglob(ufllib_dir + "/**/*.lst", recursive=True):
        with open(list_file, "r") as f:
            for line in f:
                ins_dir = os.path.dirname(list_file)
                yield os.path.join(ins_dir, line.strip()).replace("\\", "/")


def ufllib_iterator(ufllib_dir):
    """
    Iterate through all the instance inside the UflLib directory.

    Return (ins_path, num_city, num_facility, opening_cost, connection_cost).
    """
    for ins_path in ins_path_finder(ufllib_dir):
        yield (ins_path, *ins_parser(ins_path))


#%%
if __name__ == "__main__":
    print("Example Usage:")
    print(f"{'='*80}\n")
    print(f"ins_parser(ins_path): {ins_parser.__doc__}")
    print(f"{'='*40}\n")
    try:
        print(ins_parser("UflLib\\BildeKrarup\\B\\B1.2"))
    except FileNotFoundError as e:
        logger.error(e)

    print(f"{'='*80}\n")
    print(f"ins_path_finder(ufllib_dir): {ins_path_finder.__doc__}")
    print(f"{'='*40}\n")
    try:
        print("\n".join(list(ins_path_finder("UflLib"))[:5]))
        print("...")
    except OSError as e:
        logger.error(e)

    print(f"{'='*80}\n")
    print(f"ufllib_iterator(ufllib_dir): {ufllib_iterator.__doc__}")
    print(f"{'='*40}\n")
    try:
        ufllib = ufllib_iterator("UflLib")
        for i, ins in enumerate(ufllib, 1):
            print(f"Instance #{i} = {ins}\n")
            if i >= 2:
                break
    except OSError as e:
        logger.error(e)

