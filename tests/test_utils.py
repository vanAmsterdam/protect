import pytest
import jax
from jax import numpy as jnp, random
from protect.utils import time_event_to_time_cens, generate_cv_splits, generate_cv_intrain_matrix, summary_likelihoods_to_df, summary_likelihoods_to_df_per_ppmode

def test_time_event_to_time_cens():
    time = jnp.array([1, 2, 3, 4, 5])
    event = jnp.array([0, 1, 0, 1, 0])
    time_cens = time_event_to_time_cens(time, event, maxtime=4.5)
    assert time_cens.tolist() == [-1, 2, -3, 4, -4.5]

# def test_time_event_to_time_cens_no_maxtime():
#     time = jnp.array([1, 2, 3, 4, jnp.inf])
#     event = jnp.array([0, 1, 0, 1, 0])
#     with pytest.raises(ValueError):
#         time_event_to_time_cens(time, event)
# TODO: think about handling this case in a way that is jittable (e.g. with jax's checkify)


@pytest.fixture
def lls_by_fold_ppmode():
    return {
        ('train', 'obs_site1'): {
            'sum': -1*jnp.array([1., 2., 3.]),
            'mean': -1*jnp.array([1., 2., 3.])
        },
        ('train', 'obs_site2'): {
            'sum': -1*jnp.array([1., 2., 3.]),
            'mean': -1*jnp.array([1., 2., 3.])
        },
    }

@pytest.fixture
def lls_ppmode():
    return {
        ('train', 'obs_site1'): {
            'sum': jnp.array(-1.0),
            'mean': jnp.array(-2.0)
        },
        ('train', 'obs_site2'): {
            'sum': jnp.array(-1.0),
            'mean': jnp.array(-2.0)
        },
    }

def test_likelihood_df_by_fold_ppmode(lls_by_fold_ppmode):
    lldf = summary_likelihoods_to_df_per_ppmode(lls_by_fold_ppmode, dimnames=["fold"])
    assert set(lldf.columns) == {'fold', 'obs_site', 'sum', 'mean', 'split'}
    assert lldf.shape == (6, 5)
    assert lldf['fold'].unique().shape[0] == 3
    assert jnp.issubdtype(lldf['sum'].dtype, jnp.floating)
    assert jnp.issubdtype(lldf['mean'].dtype, jnp.floating)

def test_likelihood_df_ppmode(lls_ppmode):
    lldf = summary_likelihoods_to_df_per_ppmode(lls_ppmode)
    assert set(lldf.columns) == {'obs_site', 'sum', 'mean', 'split'}
    assert lldf.shape == (2, 4)
    assert jnp.issubdtype(lldf['sum'].dtype, jnp.floating)
    assert jnp.issubdtype(lldf['mean'].dtype, jnp.floating)

@pytest.fixture
def lls():
    return {
        'ppmode1': {
            ('train', 'obs_site1'): {
                'sum': jnp.array(-1.0),
                'mean': jnp.array(-2.0)
            },
            ('train', 'obs_site2'): {
                'sum': jnp.array(-1.0),
                'mean': jnp.array(-2.0)
            },
        },
        'ppmode2': {
            ('train', 'obs_site1'): {
                'sum': jnp.array(-1.0),
                'mean': jnp.array(-2.0)
            },
            ('train', 'obs_site2'): {
                'sum': jnp.array(-1.0),
                'mean': jnp.array(-2.0)
            }
        }
    }


@pytest.fixture
def lls_by_fold():
    return {
        'ppmode1': {
            ('train', 'obs_site1'): {
                'sum': -1*jnp.array([1., 2., 3.]),
                'mean': -1*jnp.array([1., 2., 3.])
            },
            ('train', 'obs_site2'): {
                'sum': -1*jnp.array([1., 2., 3.]),
                'mean': -1*jnp.array([1., 2., 3.])
            },
        },
        'ppmode2': {
            ('train', 'obs_site1'): {
                'sum': -1*jnp.array([1., 2., 3.]),
                'mean': -1*jnp.array([1., 2., 3.])
            },
            ('train', 'obs_site2'): {
                'sum': -1*jnp.array([1., 2., 3.]),
                'mean': -1*jnp.array([1., 2., 3.])
            }
        }
    }

@pytest.fixture
def lls_by_fold_and_rep():
    return {
        'ppmode1': {
            ('train', 'obs_site1'): {
                'sum': -1*jnp.array([[1., 2.], [3., 4.]]),
                'mean': -1*jnp.array([[1., 2.], [3., 4.]])
            },
            ('train', 'obs_site2'): {
                'sum': -1*jnp.array([[1., 2.], [3., 4.]]),
                'mean': -1*jnp.array([[1., 2.], [3., 4.]])
            },
        },
        'ppmode2': {
            ('train', 'obs_site1'): {
                'sum': -1*jnp.array([[1., 2.], [3., 4.]]),
                'mean': -1*jnp.array([[1., 2.], [3., 4.]])
            },
            ('train', 'obs_site2'): {
                'sum': -1*jnp.array([[1., 2.], [3., 4.]]),
                'mean': -1*jnp.array([[1., 2.], [3., 4.]])
            }
        }
    }

def test_likelihood_df_by_fold_and_rep(lls_by_fold_and_rep):
    lldf = summary_likelihoods_to_df(lls_by_fold_and_rep, dimnames=["rep", "fold"])
    assert set(lldf.columns) == {'ppmode', 'rep', 'fold', 'obs_site', 'sum', 'mean', 'split'}
    assert lldf.shape == (16, 7)
    assert lldf['fold'].unique().shape[0] == 2
    assert lldf['rep'].unique().shape[0] == 2
    assert lldf['ppmode'].unique().shape[0] == 2
    assert jnp.issubdtype(lldf['sum'].dtype, jnp.floating)
    assert jnp.issubdtype(lldf['mean'].dtype, jnp.floating)


def test_likelihood_df_by_fold(lls_by_fold):
    lldf = summary_likelihoods_to_df(lls_by_fold, dimnames=["fold"])
    assert set(lldf.columns) == {'ppmode', 'fold', 'obs_site', 'sum', 'mean', 'split'}
    assert lldf.shape == (12, 6)
    assert lldf['fold'].unique().shape[0] == 3
    assert lldf['ppmode'].unique().shape[0] == 2
    assert jnp.issubdtype(lldf['sum'].dtype, jnp.floating)
    assert jnp.issubdtype(lldf['mean'].dtype, jnp.floating)

def test_likelihood_df_by_fold_wrong_arguments(lls_by_fold):
    # assert that when not providing a dimnames argument, the function 'summary_likelihoods_to_df' throws a valueerror
    with pytest.raises(ValueError):
        summary_likelihoods_to_df(lls_by_fold, dimnames=["fold", "rep"])

    with pytest.raises(ValueError):
        summary_likelihoods_to_df(lls_by_fold)



    
def test_likelihood_df(lls):
    lldf = summary_likelihoods_to_df(lls)
    assert set(lldf.columns) == {'ppmode', 'obs_site', 'sum', 'mean', 'split'}
    assert lldf.shape == (4, 5)
    assert lldf['ppmode'].unique().shape[0] == 2
    assert jnp.issubdtype(lldf['sum'].dtype, jnp.floating)
    assert jnp.issubdtype(lldf['mean'].dtype, jnp.floating)
    

@pytest.fixture
def test_data(num_obs=10):
    return {
        'feature1': jnp.arange(num_obs),
        'feature2': 10. + jnp.arange(num_obs),
        'feature3': 50. + jnp.arange(num_obs)
    }

def test_generate_cv_splits_divisible():
    key = random.PRNGKey(0)
    num_obs = 10
    num_folds = 5
    splits = generate_cv_splits(key, num_obs, num_folds)
    
    # Check the number of splits
    assert len(splits) == num_folds
    
    # Check that each split has the correct number of training samples
    for i in range(num_folds):
        assert splits[i].shape[0] == num_obs - num_obs // num_folds
    
    # Check that all indices are covered
    all_indices = jnp.concatenate(splits)
    assert jnp.unique(all_indices).shape[0] == num_obs
    
    # Check that there are no duplicate indices within each split
    for split in splits:
        assert jnp.unique(split).shape[0] == split.shape[0]

def test_generate_cv_splits_not_divisible():
    key = random.PRNGKey(0)
    num_obs = 10
    num_folds = 3
    splits = generate_cv_splits(key, num_obs, num_folds)
    
    # Check the number of splits
    assert len(splits) == num_folds

    # Check that each split has the correct number of training samples
    for i in range(num_folds):
        assert splits[i].shape[0] == num_obs - num_obs // num_folds
    
    # Check that all indices are covered
    all_indices = jnp.concatenate(splits)
    assert jnp.unique(all_indices).shape[0] == num_obs

    # Check that the max index is num_obs - 1
    assert jnp.max(all_indices) == num_obs - 1
    
    # Check that there are no duplicate indices within each split
    for split in splits:
        assert jnp.unique(split).shape[0] == split.shape[0]

def test_generate_cv_splits_with_data(test_data):
    key = jax.random.PRNGKey(0)
    num_obs = 10
    num_folds = 3
    splits = generate_cv_splits(key, num_obs, num_folds)
    
    for split in splits:
        for key, value in test_data.items():
            train_data = value[split]
            assert not jnp.isnan(train_data).any()


def test_generate_cv_intrain_matrix():
    key = random.PRNGKey(0)
    num_obs = 10
    num_folds = 5
    splits = generate_cv_splits(key, num_obs, num_folds)
    
    # check against matrix function
    intrain_matrix = generate_cv_intrain_matrix(key, num_obs, num_folds)
    assert intrain_matrix.shape == (num_folds, num_obs)

    # check the rows of the matrix against the splits
    for i in range(num_folds):
        assert jnp.all(intrain_matrix[i, :] == jnp.isin(jnp.arange(num_obs), splits[i]))

    # check that the number of training samples is always the same
    assert jnp.all(jnp.sum(intrain_matrix, axis=1) == num_obs - num_obs // num_folds)

def test_array_to_df():
    from protect.utils import array_to_df

    # 0D test
    arr0 = jnp.array(42)
    df0 = array_to_df(arr0, [])
    assert df0.shape == (1, 1)
    assert df0.columns.tolist() == ['value']
    assert df0['value'].iloc[0] == 42

    # 0D test without specifying argument
    arr0 = jnp.array(42)
    df0 = array_to_df(arr0)
    assert df0.shape == (1, 1)
    assert df0.columns.tolist() == ['value']
    assert df0['value'].iloc[0] == 42

    # 1D test
    arr1 = jnp.array([10, 20])
    df1 = array_to_df(arr1, ['index'])
    assert df1.shape == (2, 2)
    assert 'index' in df1.columns
    assert 'value' in df1.columns
    assert df1['value'].tolist() == [10, 20]

    # 2D test
    arr2 = jnp.array([[1, 2], [3, 4]])
    df2 = array_to_df(arr2, ['row', 'col'])
    assert df2.shape == (4, 3)
    assert set(df2.columns) == {'row', 'col', 'value'}

    # 3D test
    arr3 = jnp.arange(8).reshape((2, 2, 2))
    df3 = array_to_df(arr3, ['a', 'b', 'c'])
    assert df3.shape == (8, 4)
    assert set(df3.columns) == {'a', 'b', 'c', 'value'}

