# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:29:34 2019

@author: fangzhou
"""

from formulations import canonical_formulation
from utility_ufllib import ins_parser, ufllib_iterator
import numpy as np
from loguru import logger
import sys
import os

log_name_prefix = os.path.splitext(os.path.basename(__file__))[0]
log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": "<green>{time:HH:mm:ss}</green> <level>{message}</level>",
        },
        {
            "sink": log_name_prefix + ".log",
            "format": "{time:YYYY-MM-DD HH:mm:ss} {message}",
            "level": "INFO",
        },
    ]
}
logger.configure(**log_config)


def run_tests(ufllib_dir, test, formulation, stop=None, **kwargs):
    ufllib = ufllib_iterator(ufllib_dir)
    pass_counter, ins_counter = 0, 0
    for ins_path, *ufl_instance in ufllib:
        for cardinality_param in np.linspace(1, num_facility, num=5):
            ins_counter += 1
            if test(formulation, *ufl_instance, cardinality_param, **kwargs):
                pass_counter += 1
                logger.info(
                    f"Passed ({pass_counter}/{ins_counter}): "
                    f"'{ins_path}', cardinality_param={cardinality_param}"
                )
            else:
                logger.error(
                    f"Failed ({pass_counter}/{ins_counter}): "
                    f"'{ins_path}', cardinality_param={cardinality_param}"
                )
            if stop is not None and ins_counter >= stop:
                return pass_counter, ins_counter


def sol_mip_contained_in_lp(formulation, *instance, dualized=False, purify_tol=0):
    mip_result = formulation(
        *instance,
        relaxed=False,
        dualized=dualized,
        purify_tol=purify_tol,
        OutputFlag=False,
    )
    lp_result = formulation(
        *instance,
        relaxed=True,
        dualized=dualized,
        purify_tol=purify_tol,
        OutputFlag=False,
    )
    return all(mip_result[0] <= np.ceil(lp_result[0]))


if __name__ == "__main__":

    (num_city, num_facility, opening_cost, connection_cost) = ins_parser(
        "UflLib\\BildeKrarup\\B\\B1.4"
    )
    result = sol_mip_contained_in_lp(
        canonical_formulation,
        num_city,
        num_facility,
        opening_cost,
        connection_cost,
        10,
#        dualized=True,
#        purify_tol=1e-6,
    )
    print(result)

    results = run_tests(
        "UflLib",
        sol_mip_contained_in_lp,
        canonical_formulation,
#        stop=50,
        dualized=True,
        purify_tol=1e-6,
    )
