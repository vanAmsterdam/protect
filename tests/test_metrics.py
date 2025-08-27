
import math
import jax
import pytest
import numpy as np
import jax.numpy as jnp

from causalprotect.utils import (
    harrell_c,
    harrell_c_streaming,
    roc_auc,
)


@pytest.mark.parametrize("y, s, expected_auc", [
    # Perfect separation
    (jnp.array([0,0,1,1]), jnp.array([0.1,0.2,0.8,0.9]), 1.0),
    # Complete misranking
    (jnp.array([0,0,1,1]), jnp.array([0.9,0.8,0.2,0.1]), 0.0),
    # Classic small example
    (jnp.array([0,0,1,1,0,1]), jnp.array([0.1,0.4,0.35,0.8,0.5,0.6]), 0.7777777777777778),
    # Ties average to 0.5
    (jnp.array([0,1,0,1]), jnp.array([0.5,0.5,0.2,0.2]), 0.5),
])
def test_auc_unweighted_cases(y, s, expected_auc):
    auc = float(roc_auc(y, s))
    assert auc == pytest.approx(expected_auc, rel=1e-6)

def test_auc_weighted_integer_weights():
    y = jnp.array([0,1,0,1])
    s = jnp.array([0.5,0.5,0.5,0.1])
    w = jnp.array([1,3,1,1], dtype=jnp.float32)
    auc = float(roc_auc(y, s, sample_weight=w))
    assert auc == pytest.approx(0.375, rel=1e-6)

def test_auc_weighted_mixed():
    y = jnp.array([0,0,1,1,0,1])
    s = jnp.array([0.1,0.4,0.35,0.8,0.5,0.6])
    w = jnp.array([1,2,1,1,1,3], dtype=jnp.float32)
    auc = float(roc_auc(y, s, sample_weight=w))
    assert auc == pytest.approx(0.85, rel=1e-6)

def test_degenerate_all_negative_returns_nan():
    y = jnp.array([0,0,0,0])
    s = jnp.array([0.1,0.2,0.3,0.4])
    auc = float(roc_auc(y, s))
    assert np.isnan(auc)

def test_degenerate_all_positive_returns_nan():
    y = jnp.array([1,1,1,1])
    s = jnp.array([0.1,0.2,0.3,0.4])
    auc = float(roc_auc(y, s))
    assert np.isnan(auc)

def test_batch_with_vmap():
    Ys = jnp.stack([jnp.array([0,0,1,1]), jnp.array([0,0,1,1])])
    Ss = jnp.stack([jnp.array([0.1,0.2,0.8,0.9]), jnp.array([0.9,0.8,0.2,0.1])])

    aucs = jax.vmap(roc_auc)(Ys, Ss)
    assert aucs[0] == pytest.approx(1.0)
    assert aucs[1] == pytest.approx(0.0)


## tests for survival concordance index

def test_perfect_concordance():
    T = jnp.array([1., 2., 3., 4.])
    S = jnp.array([4., 3., 2., 1.])  # higher risk earlier -> perfect
    E = jnp.array([1, 1, 1, 1], dtype=bool)
    c = float(harrell_c(T, S, E))
    assert c == pytest.approx(1.0, abs=1e-6)

def test_anti_concordance():
    T = jnp.array([1., 2., 3., 4.])
    S = jnp.array([1., 2., 3., 4.])  # lower risk earlier -> worst
    E = jnp.array([1, 1, 1, 1], dtype=bool)
    c = float(harrell_c(T, S, E))
    assert c == pytest.approx(0.0, abs=1e-6)

def test_ties_count_as_half():
    # Pairs: (0,1)=tie -> 0.5; (0,2)=1>0 ->1; (1,2)=1>0 ->1  => (0.5+1+1)/3 = 0.833333...
    T = jnp.array([1., 2., 3.])
    S = jnp.array([1., 1., 0.])
    E = jnp.array([1, 1, 1], dtype=bool)
    c = float(harrell_c(T, S, E))
    assert c == pytest.approx(2.5/3.0, abs=1e-6)

def test_censoring_rules_out_early_censored_as_defining_event():
    # i=0 is censored at time 2 -> cannot define comparable pairs
    # Only i=1 (event at 3) compares to j=2 (time 4): S[1]=0.2 > S[2]=0.1 -> concordant
    T = jnp.array([2., 3., 4.])
    E = jnp.array([0, 1, 1], dtype=bool)
    S = jnp.array([0.9, 0.2, 0.1])
    c = float(harrell_c(T, S, E))
    assert c == pytest.approx(1.0, abs=1e-6)

def test_no_comparable_pairs_returns_nan():
    # All censored -> no comparable pairs
    T = jnp.array([1., 1.])
    S = jnp.array([0.5, 0.7])
    E = jnp.array([0, 0], dtype=bool)
    c = float(harrell_c(T, S, E))
    assert math.isnan(c)

def test_streaming_matches_vectorized():
    # Deterministic small case
    T = jnp.array([5., 6., 6., 2., 4., 3.])
    E = jnp.array([1, 0, 1, 1, 0, 1], dtype=bool)
    S = jnp.array([0.2, 0.1, 0.9, 0.8, 0.6, 0.4])
    c1 = float(harrell_c(T, S, E))
    c2 = float(harrell_c_streaming(T, S, E))
    if math.isnan(c1) and math.isnan(c2):
        assert True
    else:
        assert c1 == pytest.approx(c2, abs=1e-6)

def test_shape_mismatch_raises():
    T = jnp.array([1., 2., 3.])
    S = jnp.array([0.1, 0.2])  # wrong length
    E = jnp.array([1, 1, 1], dtype=bool)
    with pytest.raises(ValueError):
        _ = harrell_c(T, S, E)

def test_survival_like_outputs_need_sign_flip():
    # Model outputs "higher=better" (e.g., survival prob). Flipping sign fixes it.
    T = jnp.array([1., 2., 3., 4.])
    survival_like = jnp.array([0.1, 0.2, 0.3, 0.4])  # higher later -> better later
    E = jnp.array([1, 1, 1, 1], dtype=bool)

    c_bad = float(harrell_c(T, survival_like, E))         # should be ~0.0
    c_good = float(harrell_c(T, -survival_like, E))       # should be ~1.0

    assert c_bad == pytest.approx(0.0, abs=1e-6)
    assert c_good == pytest.approx(1.0, abs=1e-6)
