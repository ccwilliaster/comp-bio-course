#!/usr/bin/env python
"""These are Nose-based unit tests for chemoUtils.py
"""

from nose import with_setup
from tosubmit.chemoUtils import *
logger.setLevel(10) # 10 = Debug

#.............................................................................
#   Setup functions (runs before anything)

def setup_dicts():
    drugs = { "d1" : set([0,1,2,3]),
              "d2" : set([0,1,2,3]),
              "d3" : set([3,4,5,6]),
              "d4" : set([5,6,7,8]),
              "d5" : set([9,10,11,12]) }

    drug_to_targets = { "d1" : set(["p1"]),
                        "d2" : set(["p1", "p2"]),
                        "d3" : set(["p3", "p4"]),
                        "d4" : set(["p3", "p4"]),
                        "d5" : set(["p5"]) }
    
    target_to_drugs = { "p1" : set(["d1", "d2"]),
                        "p2" : set(["d2"]),
                        "p3" : set(["d3", "d4"]), 
                        "p4" : set(["d3", "d4"]),
                        "p5" : set(["d5"]) }
    return drugs, drug_to_targets, target_to_drugs
#.............................................................................
#   Tests
def test_tanimoto_and_tanimoto_summary():
    drugs, d2t, t2d        = setup_dicts()
    sort_drugs, sort_feats = dict_to_sorted_lists(drugs, vals_as_array=True)
    tan_array              = make_tanimoto_array(sort_feats)

    rows, cols = tan_array.shape
    assert rows == cols == 5
    assert (sort_drugs[0] == "d1") and (sort_drugs[4] == "d5")
    assert tan_array[0,1] == 1
    assert ( tan_array[0,2] - (1/7.)) < 0.0000001
    assert ( tan_array[2,3] - (2/6.)) < 0.0000001
    assert tan_array[0,3] == 0

    for i in tan_array[4,0:4]: assert i == 0 # no matches except self
    for row in range(rows):
        for col in range(cols):
            # should be symmetric and diagonal should be 1
            assert tan_array[row,col] == tan_array[col,row]
            if row == col: 
                assert tan_array[row,col] == 1

    # Now test summary scores
    T1 = get_tanimoto_summary(tan_array, range(5), range(5), cutoff=0.99)
    T2 = get_tanimoto_summary(tan_array, range(5), range(5), cutoff=1)
    T3 = get_tanimoto_summary(tan_array, [0,1], [4])
    T4 = get_tanimoto_summary(tan_array, [0,1], [0,1])
    T5 = get_tanimoto_summary(tan_array, [0], [0])
    T6 = get_tanimoto_summary(tan_array, [0], [0,4])
    assert T1 == 5 + 2 # for 1's diagonal plus 2 identical scores
    assert T2 == 0     # nothing meets this cutoff criteria
    assert T3 == 0     # drug @ idx 4 has no similarity to 1 and 2
    assert T4 == 4          
    assert T5 == 1     # drug and itself == 1
    assert T6 == 1     # ditto

def test_get_drug_indices():
    drugs, d2t, t2d        = setup_dicts()
    sort_drugs, sort_feats = dict_to_sorted_lists(drugs, vals_as_array=True)
     
    idxs = get_drug_indices(sort_drugs, *[ ["d%i" % ID] for ID in range(1,6) ]) 
    assert idxs == [[i] for i in range(5)]

def test_indices_from_targets():
    drugs, d2t, t2d        = setup_dicts()
    sort_drugs, sort_feats = dict_to_sorted_lists(drugs, vals_as_array=True)
    sort_targets, drugs    = dict_to_sorted_lists(t2d)
    idxs = indices_from_targets(sort_targets, sort_drugs, t2d)
    idxs = [ sorted(idx_list) for idx_list in idxs ] # sort for comparison
    assert idxs == [ [0,1],[1],[2,3],[2,3],[4] ]

def test_share_targets():
    drugs, d2t, t2d = setup_dicts() 
    assert share_targets(d2t['d1'], d2t['d2']) == 1 
    assert share_targets(d2t['d2'], d2t['d1']) == 1 
    assert share_targets(d2t['d1'], d2t['d1']) == 1
    assert share_targets(d2t['d1'], d2t['d3']) == 0
    assert share_targets(d2t['d3'], d2t['d4']) == 1
    assert share_targets(d2t['d1'], d2t['d5']) == 0

def test_get_random_indices():
    T1 = get_random_indices((1,2,3,4), [1 for i in range(10)])
    T2 = get_random_indices((1,1,1,1,1), [1 for i in range(10)])
    T3 = get_random_indices((1,), [100])
    T4 = get_random_indices((0,), [0])

    assert T1 == [ [1], [1,1], [1,1,1], [1,1,1,1] ]
    assert T2 == [ [1], [1], [1], [1], [1] ]
    assert T3 == [ [100] ]
    assert T4 == [ [ ] ]
