# Stochastic-BFL

Need to install `loguru`:

```
pip install loguru
```

**Note**: All python modules come with a demo usage under `if __name__ == "__main__":`

## `utility_ufllib.py`

`ins_parser(ins_path)`:

- Parse an UflLib instance file given the path to that instance
- Input: file path to standard UflLib-format data.
- Format description see http://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib/
- Return `num_city, num_facility, opening_cost, connection_cost`


`ins_path_finder(ufllib_dir)`:

- Iteratively find all instance paths inside the UflLib directory.
- Return instance path.

`ufllib_iterator(ufllib_dir)`:

- Iterate through all the instance inside the UflLib directory.
- Return `ins_path, num_city, num_facility, opening_cost, connection_cost`

## `formulations.py`

`canonical_formulation(
    num_city,
    num_facility,
    opening_cost,
    connection_cost,
    cardinality_param,
    dualized=False,
    relaxed=False,
    purify_tol=0,
    OutputFlag=False):`

- Return `opening_, alloc_, dual_card` if `relaxed and not dualized`; otherwise, `opening_, alloc_`
- `dual_card` is dual value of the cardinality constraint.

`new_formulation(
    num_city,
    num_facility,
    opening_cost,
    connection_cost,
    cardinality_param,
    dualized=False,
    relaxed=False,
    purify_tol=0,
    OutputFlag=False):`

- Return the same as `canonical_formulation()`

## `test.py`

`sol_mip_contained_in_lp(formulation, *instance, dualized=False, purify_tol=0)`:

- returns single test pass result 
- `True` if MIP opening solution is contained in ceiled LP opening solution; `False` otherwise

`run_tests(ufllib_dir, test, formulation, stop=None, **kwargs)`:

- runs tests (specified by `test` method) on all instance files found in `ufllib_dir`, using a fomulation (specified by `formulation` method), 
- stop after `stop` number of instances
- returns `pass_counter, ins_counter`
