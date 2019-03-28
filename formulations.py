# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:22:01 2019

@author: fangzhou
"""


from gurobipy import *
import numpy as np
from loguru import logger
from utility_ufllib import ins_parser


def canonical_formulation(
    num_city,
    num_facility,
    opening_cost,
    connection_cost,
    cardinality_param,
    dualized=False,
    relaxed=False,
    purify_tol=0,
    OutputFlag=False
):
    Cities = np.arange(num_city)
    Facilities = np.arange(num_facility)

    m = Model("canonical_formulation")
    alloc = m.addVars(Cities, Facilities, name="alloc")
    if relaxed:
        opening = m.addVars(Facilities, name="opening")
    else:
        opening = m.addVars(Facilities, vtype="B", name="opening")
    m.addConstrs((alloc.sum(i, "*") == 1 for i in Cities), name="ct1")
    m.addConstrs(
        (alloc[i, j] <= opening[j] for i in Cities for j in Facilities), name="ct2"
    )

    obj = LinExpr(connection_cost.flatten(), alloc.values())
#    obj += LinExpr(opening_cost, opening.values())
    if dualized:
        obj += cardinality_param * opening.sum()
    else:
        ct_card = m.addConstr(opening.sum() <= cardinality_param, name="ct_card")
    m.setObjective(obj, GRB.MINIMIZE)

    m.params.OutputFlag = OutputFlag
    m.optimize()

    opening_ = np.abs(m.getAttr("X", opening.values()))
    alloc_ = np.abs(m.getAttr("X", alloc.values())).reshape(num_city, num_facility)
    opening_[opening_ < purify_tol] = 0
    if relaxed and not dualized:
        dual_card = ct_card.Pi
        return opening_, alloc_, dual_card
    else:
        return opening_, alloc_


def new_formulation(
    num_city,
    num_facility,
    opening_cost,
    connection_cost,
    cardinality_param,
    dualized=False,
    relaxed=False,
    purify_tol=0,
    OutputFlag=False
):
    Cities = np.arange(num_city)
    Facilities = np.arange(num_facility)
    connection_cost_sorted = np.sort(connection_cost, axis=1)
    D = [np.unique(connection_cost_sorted[i]) for i in Cities]
    K = [np.arange(len(D[i])) for i in Cities]
    z_index_set = [(i, k) for i in Cities for k in K[i]]

    nf = Model("new_formulation")
    z = nf.addVars(z_index_set, vtype="C", name="z")
    if relaxed:
        opening = nf.addVars(Facilities, name="opening")
    else:
        opening = nf.addVars(Facilities, vtype="B", name="opening")

    nf.addConstrs(
        (
            z[i, k]
            + quicksum(opening[j] for j in np.where(connection_cost[i] == D[i][k])[0])
            >= (1 if k == 0 else z[i, k - 1])
            for i in Cities
            for k in K[i]
        ),
        name="c17_c18",
    )
    nf.addConstrs((z[i, K[i][-1]] == 0 for i in Cities), name="c19")


    obj = connection_cost_sorted[:, 0].sum() + quicksum(
        (D[i][k + 1] - D[i][k]) * z[i, k] for i in Cities for k in K[i][:-1]
    )
#    obj += LinExpr(opening_cost, opening.values())
    if dualized:
        obj += cardinality_param * opening.sum()
    else:
        ct_card = nf.addConstr(opening.sum() <= cardinality_param, name="ct_card")
    nf.setObjective(obj, GRB.MINIMIZE)

    nf.params.OutputFlag = OutputFlag
    nf.optimize()

    opening_ = np.abs(nf.getAttr("X", opening.values()))
    z_ = np.full((num_city, num_facility), np.nan)
    for i in Cities:
        z_[i, K[i]] = np.abs(nf.getAttr("X", z.select(i)))
    opening_[opening_ < purify_tol] = 0
    if relaxed and not dualized:
        dual_card = ct_card.Pi
        return opening_, z_, dual_card
    else:
        return opening_, z_


#%%
if __name__ == "__main__":
    print("Example Usage:")
    print(f"{'='*80}\n")
    try:
        (num_city, num_facility, opening_cost, connection_cost) = ins_parser(
            "UflLib\\BildeKrarup\\B\\B1.4"
        )
        cardinality_param = 5
        result = canonical_formulation(
            num_city,
            num_facility,
            opening_cost,
            connection_cost,
            cardinality_param,
#            dualized=True,
            relaxed=True,
#            purify_tol=1e-6,
            OutputFlag=True
        )
        print(f"{'='*80}\n")
        print("Canonical formulation result:\n")
        print(result)
        print(f"{'='*80}\n")

        result2 = new_formulation(
            num_city,
            num_facility,
            opening_cost,
            connection_cost,
            cardinality_param,
#            dualized=True,
            relaxed=True,
            purify_tol=1e-6,
            OutputFlag=True
        )
        print(f"{'='*80}\n")
        print("New formulation result:\n")
        print(result2)
        print(f"{'='*80}\n")
    except FileNotFoundError as e:
        logger.error(e)
