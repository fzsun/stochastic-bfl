# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:29:34 2019

@author: fangzhou
"""

from formulations import canonical_formulation, new_formulation
from utility_ufllib import ins_parser, ufllib_iterator
import numpy as np
from scipy import stats
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


def run_tests(ufllib_dir, test, formulation, seed=None, stop=None, **kwargs):
    ufllib = ufllib_iterator(ufllib_dir)
    np.random.seed(seed)
    pass_counter, ins_counter = 0, 0
    for ins_path, *ufl_instance in ufllib:
        num_facility = ufl_instance[1]
        card_param_set = [1, num_facility]
        card_param_set[1:1] =  np.random.uniform(1, num_facility, 3).round(3)
        card_param_set = sorted(card_param_set)
        for cardinality_param in card_param_set:
            ins_counter += 1
            if test(formulation, *ufl_instance, cardinality_param, **kwargs):
                pass_counter += 1
                logger.info(
                    f"Passed ({pass_counter}/{ins_counter}): "
                    f"'{ins_path}', cardinality_param={cardinality_param:>8.3f}"
                )
            else:
                logger.error(
                    f"Failed ({pass_counter}/{ins_counter}): "
                    f"'{ins_path}', cardinality_param={cardinality_param:>8.3f}"
                )
            if stop is not None and ins_counter >= stop:
                return pass_counter, ins_counter

def dual_match_lambda(formulation, *instance, obj_diff_tol=0.01):
    m_model_result = formulation(*instance, relaxed=True, dualized=False)
    dual = -m_model_result[-1]
    lam_model_result = formulation(*(instance[:-1]), dual, relaxed=True, dualized=True)
    m_obj = np.sum(m_model_result[1] * instance[3])
    lam_obj = np.sum(lam_model_result[1] * instance[3])
    obj_diff = np.abs(m_obj - lam_obj) / m_obj
    if obj_diff <  obj_diff_tol:
        return True
    else:
        logger.error(f"Obj_diff: {obj_diff:.2%}")
        return False

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
    try:
        instance = (*ins_parser("UflLib/BildeKrarup/B/B1.4"), 10)
        test_result = dual_match_lambda(canonical_formulation, *instance)
        logger.info(f"dual_match_lambda Passed test = {test_result}")
        instance = (*ins_parser('UflLib/BildeKrarup/B/B1.7'), 25.5)
        test_result = sol_mip_contained_in_lp(canonical_formulation, *instance)
        logger.info(f"sol_mip_contained_in_lp Passed test = {test_result}")
#        mip_result = canonical_formulation(*instance, relaxed=False, OutputFlag=True)
#        lp_result = canonical_formulation(*instance, relaxed=True, OutputFlag=True)
#        mip_result = new_formulation(*instance, relaxed=False, OutputFlag=True)
#        lp_result = new_formulation(*instance, relaxed=True, OutputFlag=True)
    except OSError as e:
        logger.error(e)

    try:
        results = run_tests(
            "UflLib",
#            sol_mip_contained_in_lp,
            dual_match_lambda,
            canonical_formulation,
            seed=1
#            stop=50,
    #        dualized=True,
    #        purify_tol=1e-6,
        )
    except OSError as e:
        logger.error(e)


