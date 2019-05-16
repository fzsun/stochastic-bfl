# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:45:20 2019

@author: fangzhou

TODO:
    1. Accelerate Benders using one-tree Benders (lazy constraint callback).
    2. Apply parallelism for subproblem.
"""

from gurobipy import *
import numpy as np
from utility_ufllib import ins_parser, ufllib_iterator
from loguru import logger

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


def mean_problem(Cities, Facilities, opening_cost, probabilities, scenarios):
    """
    no stochasticity, all use mean (or nominal) parameter values
    """
    mean_connection_cost = (probabilities[:, None, None] * scenarios).sum(axis=0)

    mp = Model("mean_problem")
    alloc = mp.addVars(Cities, Facilities, name="alloc")
    opening = mp.addVars(Facilities, vtype="B", name="opening")
    mp.addConstrs((alloc.sum(i, "*") == 1 for i in Cities), name="ct1")
    mp.addConstrs(
        (alloc[i, j] <= opening[j] for i in Cities for j in Facilities), name="ct2"
    )

    obj_connection = LinExpr(mean_connection_cost.flatten(), alloc.values())
    obj_opening = LinExpr(opening_cost, opening.values())
    obj = obj_connection + obj_opening
    mp.setObjective(obj, GRB.MINIMIZE)
    mp.optimize()
    opening_ = np.abs(mp.getAttr("X", opening.values()))
    alloc_ = np.abs(mp.getAttr("X", alloc.values())).reshape(num_city, num_facility)
    return mp.objVal, opening_, alloc_


def get_VaR(probabilities, values, alpha):
    sorted_idx = values.argsort()
    prob_sort = probabilities[sorted_idx]
    cum_prob = prob_sort.cumsum()
    VaR_idx = sorted_idx[np.where(cum_prob >= alpha)[0][0]]
    VaR = values[VaR_idx]
    return VaR


def scenario_reduction(val_history, probabilities, alpha, agg_method="max"):
    #   TODO: other agg_method could be: median, mean, percentXX (XX = 01 to 99)
    if agg_method == "max":
        aggregated = np.max(val_history, axis=0)
        scn_to_remove = aggregated.argsort()[0]
    VaR = get_VaR(probabilities, aggregated, alpha)
    if aggregated[scn_to_remove] > VaR:
        logger.warning("Cannot remove scenario since its value > VaR.")
        return probabilities, alpha
    prob_new = probabilities.copy()
    prob_new[scn_to_remove] = 0
    prob_new = prob_new / prob_new.sum()
    alpha_new = (alpha - probabilities[scn_to_remove]) / (
        1 - probabilities[scn_to_remove]
    )
    return prob_new, alpha_new


def master_cutloop(
    Cities,
    Facilities,
    opening_cost,
    val_history,
    probabilities,
    scenarios,
    alpha=0.95,
    nitr=5
):
    lb, ub = 0.0, np.inf
    prob_new = probabilities.copy()
    alpha_new = alpha

    m = Model("master")
    opening = m.addVars(Facilities, vtype="B", name="opening")
    v = m.addVars(len(scenarios), name="v")
    eta = m.addVar(name="eta")

    m.addConstr(opening.sum() >= 1)

    obj_opening = LinExpr(opening_cost, opening.values())
    obj_recourse = LinExpr(probabilities / (1 - alpha), v.values())
    obj = obj_opening + eta + obj_recourse
    m.setObjective(obj, GRB.MINIMIZE)

    for itr in range(nitr):
        m.optimize()
        lb = max(lb, m.objVal)
        opening_ = np.abs(m.getAttr("X", opening.values()))
        eta_ = eta.x
        r_objs, allocs_, dual_ct1s, dual_ct2s = loop_scenario(
            Cities, Facilities, opening_, prob_new, scenarios
        )
        # There are two versions of actual_total_obj, do both work? 2nd seems better.
        # 1st version
        # actual_total_obj = (
        # opening_cost @ opening_
        #     + eta_
        #     + probabilities / (1 - alpha) @ (r_objs - eta_).clip(0)
        # )
        # 2nd version
        VaR = get_VaR(prob_new, r_objs, alpha_new)
        actual_total_obj = (
            opening_cost @ opening_
            + VaR
            + prob_new / (1 - alpha_new) @ (r_objs - VaR).clip(0)
        )
        ub = min(ub, actual_total_obj)

        val_history = np.concatenate((val_history, r_objs[None]), axis=0)
        prob_new, alpha_new = scenario_reduction(
            val_history, prob_new, alpha_new, agg_method="max"
        )
        # TODO: revise model accordingly
        # Add Benders cuts
        for i in range(len(scenarios)):
            if prob_new[i] == 0:
                v[i].obj = 0  # effectively, this scenario is removed
            else:
                rhs = dual_ct1s[i].sum() + LinExpr(
                    dual_ct2s[i].sum(axis=0), opening.values()
                )
                m.addConstr(v[i] + eta >= rhs)
        num_scns = len(prob_new.nonzero()[0])
        logger.info(f"Cutloop#{itr} #scn={num_scns}, alpha_new={alpha_new:.3f} VaR={VaR:.1f}, actual_obj={actual_total_obj:.1f} lb={lb:.1f} ub={ub:.1f}")
def loop_scenario(Cities, Facilities, opening_, probabilities, scenarios):
    r_objs = np.empty(len(scenarios))
    allocs_ = np.empty_like(scenarios)
    dual_ct1s = np.empty((len(scenarios), len(Cities)))
    dual_ct2s = np.empty_like(scenarios)

    for i, scenario in enumerate(scenarios):
        if probabilities[i] == 0:  # No need to solve recourse if prob == 0
            r_objs[i] = 1e100
        else:
            r_objs[i], allocs_[i], dual_ct1s[i], dual_ct2s[i] = recourse(
                Cities, Facilities, opening_, scenario
            )
        logger.info(f"recourse #{i} obj = {r_objs[i]}")
    return r_objs, allocs_, dual_ct1s, dual_ct2s


def recourse(Cities, Facilities, opening_, scenario):
    r = Model("recourse")
    alloc = r.addVars(Cities, Facilities, name="alloc")
    obj = LinExpr(scenario.flatten(), alloc.values())
    r.setObjective(obj, GRB.MINIMIZE)

    ct1 = r.addConstrs((alloc.sum(i, "*") == 1 for i in Cities), name="ct1")
    ct2 = r.addConstrs(
        (alloc[i, j] <= opening_[j] for i in Cities for j in Facilities), name="ct2"
    )

    r.params.OutputFlag = 0
    r.optimize()

    alloc_ = np.abs(r.getAttr("X", alloc.values())).reshape(len(Cities), -1)
    dual_ct1 = np.array(r.getAttr("Pi", ct1.values()))
    dual_ct2 = np.reshape(r.getAttr("Pi", ct2.values()), (len(Cities), -1))

    dual_obj = dual_ct1.sum() + (dual_ct2 * opening_).sum()
    assert abs(dual_obj - r.objVal) < 1e-6

    return r.objVal, alloc_, dual_ct1, dual_ct2


if __name__ == "__main__":

    num_city, num_facility, opening_cost, connection_cost = ins_parser(
        "UflLib/BildeKrarup/B/B1.4"
    )
    Cities = np.arange(num_city)
    Facilities = np.arange(num_facility)

    num_scenario = 123
    demand_samples = np.clip(
        np.random.normal(1, 0.2, (num_scenario, num_city)), 0, None
    )
    scenarios = demand_samples[:, :, None] * connection_cost[None]
    probabilities = np.ones(num_scenario) / num_scenario
    val_history = np.zeros_like(probabilities)[None]
    master_cutloop(
        Cities,
        Facilities,
        opening_cost,
        val_history,
        probabilities,
        scenarios,
        alpha=0.95,
        nitr=5
    )
