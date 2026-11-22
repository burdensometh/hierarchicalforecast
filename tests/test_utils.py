import numpy as np
import pandas as pd
import polars as pl
import pytest
from statsforecast.utils import generate_series
from utilsforecast.data import generate_series as util_generate_series

from hierarchicalforecast.utils import (
    CodeTimer,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
    aggregate,
    aggregate_temporal,
    aggregate_temporal_calendar,
    get_cross_temporal_tags,
    level_to_outputs,
    make_future_dataframe,
    quantiles_to_outputs,
    samples_to_quantiles_df,
)

from .conftest import assert_raises_with_message


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            'cat1': ['a', 'a', 'a', 'b'],
            'cat2': ['1', '2', '3', '2'],
            'y': [10, 20, 30, 40],
            'ds': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-02-01']
        }
    )
    df['country'] = 'COUNTRY'
    return df

@pytest.fixture
def df_pl(df):
    return pl.from_pandas(df)

@pytest.fixture
def spec():
    spec = [['country'], ['country', 'cat1'], ['country', 'cat1', 'cat2']]
    return spec

# simple case
def test_simple_case(df, spec):
    Y_df, S_df, tags = aggregate(df, spec)
    assert list(Y_df["unique_id"]) == (3 * ['COUNTRY'] + 3 * ['COUNTRY/a'] + ['COUNTRY/b'] + ['COUNTRY/a/1', 'COUNTRY/a/2', 'COUNTRY/a/3'] + ['COUNTRY/b/2'])

    assert Y_df.query("unique_id == 'COUNTRY'")['y'].tolist() == [10, 60, 30]
    assert list(S_df["unique_id"]) == ['COUNTRY', 'COUNTRY/a', 'COUNTRY/b', 'COUNTRY/a/1', 'COUNTRY/a/2', 'COUNTRY/a/3', 'COUNTRY/b/2']

    assert S_df.columns.tolist() == ['unique_id', 'COUNTRY/a/1', 'COUNTRY/a/2', 'COUNTRY/a/3', 'COUNTRY/b/2']

    expected_tags = {
        'country': ['COUNTRY'],
        'country/cat1': ['COUNTRY/a', 'COUNTRY/b'],
        'country/cat1/cat2': ['COUNTRY/a/1', 'COUNTRY/a/2', 'COUNTRY/a/3','COUNTRY/b/2'],
    }
    for k, actual in tags.items():
        assert actual.tolist() == expected_tags[k]

    # test categoricals don't produce all combinations
    df2 = df.copy()
    for col in ('country', 'cat1', 'cat2'):
        df2[col] = df2[col].astype('category')

    Y_df2, *_ = aggregate(df2, spec)
    assert Y_df.shape[0] == Y_df2.shape[0]


# Testing equivalence of other dataframe libs to pandas results.
# TODO: extend for other frameworks
def assert_eq_agg_dataframe(df, spec, frameworks=None):
    if frameworks is None:
        frameworks = ["polars"]
    for framework in frameworks:
        if framework == "polars":
            df_f = pl.from_pandas(df)
        else:
            raise ValueError(f"Framework {framework} not recognized")

        Y_df, S_df, tags = aggregate(df=df, spec=spec)
        Y_df_f, S_df_f, tags_f = aggregate(df=df_f, spec=spec)

        pd.testing.assert_frame_equal(Y_df, Y_df_f.to_pandas())
        pd.testing.assert_frame_equal(S_df, S_df_f.to_pandas())
        for tag in tags:
            np.testing.assert_equal(tags[tag], tags_f[tag])


def test_unbalanced_dataset():
    # test unbalanced dataset
    max_tenure = 24
    dates = pd.date_range(start='2019-01-31', freq='ME', periods=max_tenure)
    cohort_tenure = [24, 23, 22, 21]

    ts_list = []

    # Create ts for each cohort
    for i in range(len(cohort_tenure)):
        ts_list.append(
            generate_series(n_series=1, freq='M', min_length=cohort_tenure[i], max_length=cohort_tenure[i]).reset_index() \
                .assign(ult=i) \
                .assign(ds=dates[-cohort_tenure[i]:]) \
                .drop(columns=['unique_id'])
        )
    df = pd.concat(ts_list, ignore_index=True)

    # Create categories
    df['pen'] = np.where(df['ult'] < 2, 'a', 'b')
    # Note that unique id requires strings
    df['ult'] = df['ult'].astype(str)

    hier_levels = [
        ['pen'],
        ['pen', 'ult'],
    ]
    assert_eq_agg_dataframe(df, hier_levels)


def test_tourism_df_non_null_grouped(tourism_df, hiers_grouped, hiers_strictly):
    df = tourism_df
    # test strict
    hier_df, S_df, tags = aggregate(df=df, spec=hiers_strictly)
    assert len(hier_df) == 6800
    assert hier_df["unique_id"].nunique() == 85
    assert S_df.shape == (85, 77)
    np.testing.assert_array_equal(hier_df["unique_id"].unique(), S_df["unique_id"])
    assert len(tags) == len(hiers_strictly)


    # test grouped
    hier_df, S_df, tags = aggregate(df=df, spec=hiers_grouped)
    assert len(hier_df) == 34_000
    assert hier_df["unique_id"].nunique() == 425
    assert S_df.shape == (425, 305)
    np.testing.assert_array_equal(hier_df["unique_id"].unique(), S_df["unique_id"])
    assert len(tags) == len(hiers_grouped)
    # polars
    assert_eq_agg_dataframe(df, hiers_strictly)
    assert_eq_agg_dataframe(df, hiers_grouped)

def test_tourism_df_null_grouped(tourism_df, hiers_strictly):
    df = tourism_df
    #Unit Test NaN Values
    df_nan = df.copy()
    df_nan.loc[0, 'Region'] = float('nan')
    assert_raises_with_message(
        aggregate,
        'null values',
        df_nan, hiers_strictly,
    )

    #Unit Test None Values
    df_none = df.copy()
    df_none.loc[0, 'Region'] = None
    assert_raises_with_message(
        aggregate,
        'null values',
        df_none, hiers_strictly,
    )
    # polars

    #Unit Test NaN Values
    df_nan_pl = pl.DataFrame(df_nan)
    assert_raises_with_message(
        aggregate,
        'null values',
        df_nan_pl, hiers_strictly,
    )

    #Unit Test None Values
    df_none_pl = pl.DataFrame(df_none)
    assert_raises_with_message(
        aggregate,
        'null values',
        df_none_pl, hiers_strictly,
    )

def test_equality_sparse_non_sparse(tourism_df, hiers_strictly, hiers_grouped):
    from hierarchicalforecast.utils import SMatrix

    df = tourism_df

    # Test equality of sparse (SMatrix) and non-sparse (DataFrame) aggregation
    with CodeTimer('strict non-sparse aggregate'):
        Y_df, S_df, tags = aggregate(df=df, sparse_s=False, spec=hiers_strictly)

    with CodeTimer('strict sparse aggregate'):
        Y_df_sparse, S_df_sparse, tags_sparse = aggregate(df=df, sparse_s=True, spec=hiers_strictly)

    assert isinstance(S_df_sparse, SMatrix)
    np.testing.assert_almost_equal(Y_df.y.values, Y_df_sparse.y.values)
    np.testing.assert_array_equal(S_df.values, S_df_sparse.to_frame().values)
    np.testing.assert_array_equal(S_df.columns, S_df_sparse.to_frame().columns)

    np.testing.assert_array_equal(Y_df.columns, Y_df_sparse.columns)
    np.testing.assert_array_equal(Y_df.index, Y_df_sparse.index)

    with CodeTimer('grouped non-sparse aggregate'):
        Y_df, S_df, tags = aggregate(df=df, sparse_s=False, spec=hiers_grouped)

    with CodeTimer('grouped sparse aggregate'):
        Y_df_sparse, S_df_sparse, tags_sparse = aggregate(df=df, sparse_s=True, spec=hiers_grouped)

    assert isinstance(S_df_sparse, SMatrix)
    np.testing.assert_almost_equal(Y_df.y.values, Y_df_sparse.y.values)
    np.testing.assert_array_equal(S_df.values, S_df_sparse.to_frame().values)
    np.testing.assert_array_equal(S_df.columns, S_df_sparse.to_frame().columns)

    np.testing.assert_array_equal(Y_df.columns, Y_df_sparse.columns)
    np.testing.assert_array_equal(Y_df.index, Y_df_sparse.index)


def test_sparse_s_with_polars(tourism_df, hiers_strictly):
    """sparse_s=True should work with Polars input (previously raised an error)."""
    from hierarchicalforecast.utils import SMatrix

    df_pl = pl.from_pandas(tourism_df)
    Y_df, S_df, tags = aggregate(df=df_pl, sparse_s=True, spec=hiers_strictly)
    assert isinstance(S_df, SMatrix)
    assert S_df.shape == (85, 76)
    assert S_df.frame_shape == (85, 77)
    assert len(S_df.row_labels) == 85
    assert len(S_df.col_labels) == 76

    # Verify values match the dense Pandas path
    Y_df_pd, S_df_pd, _ = aggregate(df=tourism_df, sparse_s=False, spec=hiers_strictly)
    np.testing.assert_array_equal(
        S_df_pd.values, S_df.to_frame(backend="pandas").values
    )


@pytest.fixture
def df_for_other_cases():
    df = pd.DataFrame(
        {
            'cat1': 8 * ['a'] + 8 * ['b'],
            'y': np.linspace(10, 160, 16),
            'ds': 2 * ['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01', '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01']
        }
    )
    df['country'] = 'COUNTRY'
    df["ds"] = pd.to_datetime(df["ds"])
    df["unique_id"] = df["country"] + "/" + df["cat1"]
    return df

def test_other_simple_case(df_for_other_cases):
    df = df_for_other_cases
    # simple case
    spec = {"year": 4,
            "semiannual": 2,
            "quarter": 1}

    Y_df_te, S_df_te, tags_te = aggregate_temporal(df, spec)

    np.testing.assert_array_equal(
        list(Y_df_te["unique_id"]),
            ['COUNTRY/a', 'COUNTRY/a', 'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/a',
        'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/b', 'COUNTRY/b',
        'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/a',
        'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/a', 'COUNTRY/a',
        'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/b',
        'COUNTRY/b', 'COUNTRY/b', 'COUNTRY/b'])
    np.testing.assert_array_equal(
        list(Y_df_te["temporal_id"]),
            ['year-1', 'year-2', 'year-1', 'year-2', 'semiannual-1', 'semiannual-2', 'semiannual-3', 'semiannual-4', 'semiannual-1', 'semiannual-2', 'semiannual-3', 'semiannual-4', 'quarter-1', 'quarter-2', 'quarter-3', 'quarter-4', 'quarter-5', 'quarter-6', 'quarter-7', 'quarter-8', 'quarter-1', 'quarter-2', 'quarter-3', 'quarter-4', 'quarter-5', 'quarter-6', 'quarter-7', 'quarter-8']
    )
    np.testing.assert_array_equal(Y_df_te.query("unique_id == 'COUNTRY/a' & temporal_id in ['year-1', 'year-2']")['y'].values, [100, 260])
    np.testing.assert_array_equal(
        list(S_df_te["temporal_id"]),
        ['year-1', 'year-2', 'semiannual-1', 'semiannual-2', 'semiannual-3', 'semiannual-4', 'quarter-1', 'quarter-2', 'quarter-3', 'quarter-4',
    'quarter-5', 'quarter-6', 'quarter-7', 'quarter-8'],
    )
    np.testing.assert_array_equal( S_df_te.columns,
        ['temporal_id', 'quarter-1',
        'quarter-2', 'quarter-3',
        'quarter-4', 'quarter-5',
        'quarter-6', 'quarter-7',
        'quarter-8'],
    )
    expected_tags = {
        'year': ['year-1', 'year-2'],
        'semiannual': ['semiannual-1', 'semiannual-2', 'semiannual-3', 'semiannual-4'],
        'quarter': ['quarter-1', 'quarter-2', 'quarter-3', 'quarter-4', 'quarter-5', 'quarter-6', 'quarter-7', 'quarter-8'],
    }
    for k, actual in tags_te.items():
        np.testing.assert_array_equal(actual, expected_tags[k])


def test_future_make_df(df_for_other_cases):
    df = df_for_other_cases
    # Test make_future_dataframe
    futr_df = make_future_dataframe(df, freq='QS', h=4)

    futr_df_check = pd.DataFrame(
        {
            'unique_id': 4 * ['COUNTRY/a'] + 4 *['COUNTRY/b'],
            'ds': pd.concat(2 * [pd.date_range(start='2022-01-01', freq='QS', periods=4).to_series()], ignore_index=True),
        }
    )

    pd.testing.assert_frame_equal(futr_df, futr_df_check)
    # polars
    df_pl = pl.DataFrame(df)
    futr_df_pl = make_future_dataframe(df_pl, freq='1q', h=4)
    pd.testing.assert_frame_equal(futr_df, futr_df_pl.to_pandas())

def test_cross_temporal_tags():
    # Test cross-temporal tags
    df = pd.DataFrame(
        {
            'cat1': 8 * ['a'] + 8 * ['b'],
            'y': np.linspace(10, 160, 16),
            'ds': 2 * ['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01', '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01']
        }
    )
    df['country'] = 'COUNTRY'
    df["ds"] = pd.to_datetime(df["ds"])

    # Aggregate cross-sectionally
    spec_cs =  [['country'], ['country', 'cat1']]
    Y_df_cs, S_df_cs, tags_cs = aggregate(df, spec_cs)

    # Aggregate temporally
    spec_te = {"year": 4, "quarter": 1}
    Y_df_te, S_df_te, tags_te = aggregate_temporal(Y_df_cs, spec_te)

    # Get cross-temporal tags
    cross_df, tags_ct = get_cross_temporal_tags(Y_df_te, tags_cs, tags_te)

    tags_ct_check = {'country//year': ['COUNTRY//year-1', 'COUNTRY//year-2'],
    'country//quarter': ['COUNTRY//quarter-1',
    'COUNTRY//quarter-2',
    'COUNTRY//quarter-3',
    'COUNTRY//quarter-4',
    'COUNTRY//quarter-5',
    'COUNTRY//quarter-6',
    'COUNTRY//quarter-7',
    'COUNTRY//quarter-8'],
    'country/cat1//year': ['COUNTRY/a//year-1',
    'COUNTRY/a//year-2',
    'COUNTRY/b//year-1',
    'COUNTRY/b//year-2'],
    'country/cat1//quarter': ['COUNTRY/a//quarter-1',
    'COUNTRY/a//quarter-2',
    'COUNTRY/a//quarter-3',
    'COUNTRY/a//quarter-4',
    'COUNTRY/a//quarter-5',
    'COUNTRY/a//quarter-6',
    'COUNTRY/a//quarter-7',
    'COUNTRY/a//quarter-8',
    'COUNTRY/b//quarter-1',
    'COUNTRY/b//quarter-2',
    'COUNTRY/b//quarter-3',
    'COUNTRY/b//quarter-4',
    'COUNTRY/b//quarter-5',
    'COUNTRY/b//quarter-6',
    'COUNTRY/b//quarter-7',
    'COUNTRY/b//quarter-8']}

    for key, value in tags_ct.items():
        np.testing.assert_array_equal(value, tags_ct_check[key])
    # polars
    df_pl = pl.DataFrame(df)
    Y_df_cs_pl, S_df_cs_pl, tags_cs_pl = aggregate(df_pl, spec_cs)
    Y_df_te_pl, S_df_te_pl, tags_te_pl = aggregate_temporal(Y_df_cs_pl, spec_te)
    cross_df_pl, tags_ct_pl = get_cross_temporal_tags(Y_df_te_pl, tags_cs_pl, tags_te_pl)

    for key, value in tags_ct_pl.items():
        np.testing.assert_array_equal(value, tags_ct_check[key])


def test_temporal_agg():
    # Test temporal aggregation with synthetic series of varying length
    freq1 = "D"
    df1 = util_generate_series(n_series=2, freq=freq1, min_length=2 * 365, max_length=4 * 365, static_as_categorical=False, equal_ends=True)

    spec1  = {"year": 365, "quarter": 91, "month": 30, "week": 7, "day": 1}

    Y_df1_te, S_df1_te, tags1_te = aggregate_temporal(df1, spec1)

    assert Y_df1_te.isnull().sum().sum() == 0, "There are NaN values in the aggregated DataFrame"

    freq2 = "ME"
    df2 = util_generate_series(n_series=2, freq=freq2, min_length=24, max_length=48, static_as_categorical=False, equal_ends=True)

    spec2  = {"year": 12, "quarter": 4, "month": 1}

    Y_df2_te, S_df2_te, tags2_te = aggregate_temporal(df2, spec2)

    assert Y_df2_te.isnull().sum().sum() == 0, "There are NaN values in the aggregated DataFrame"
    # polars
    df1_pl = pl.from_pandas(df1)
    df2_pl = pl.from_pandas(df2)

    Y_df1_te_pl, S_df1_te_pl, tags1_te_pl = aggregate_temporal(df1_pl, spec1)
    Y_df2_te_pl, S_df2_te_pl, tags2_te_pl = aggregate_temporal(df2_pl, spec2)

    # Polars adds nanoseconds to the datetime columns, so we remove them for comparison
    Y_df1_te_pl_pd = Y_df1_te_pl.to_pandas()
    Y_df2_te_pl_pd = Y_df2_te_pl.to_pandas()

    pd.testing.assert_frame_equal(Y_df1_te, Y_df1_te_pl_pd)
    pd.testing.assert_frame_equal(Y_df2_te, Y_df2_te_pl_pd)


@pytest.mark.parametrize("levels, expected_quantiles, expected_names", [
    ([80, 90], [0.5, 0.05, 0.1, 0.9, 0.95], ['-median', '-lo-90', '-lo-80', '-hi-80', '-hi-90']),
    ([30], [0.5, 0.35, 0.65], ['-median', '-lo-30', '-hi-30'])
])
def test_level_to_outputs(levels, expected_quantiles, expected_names):
    result_quantiles, result_names = level_to_outputs(levels)
    np.testing.assert_array_equal(result_quantiles, expected_quantiles)
    np.testing.assert_array_equal(result_names, expected_names)


@pytest.mark.parametrize("quantiles, expected_quantiles, expected_names", [
    ([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], ['-lo-60.0', '-lo-20.0', '-hi-20.0', '-hi-60.0']),
    ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ['-lo-80.0', '-lo-60.0', '-lo-40.0', '-lo-20.0', '-median', '-hi-20.0', '-hi-40.0', '-hi-60.0', '-hi-80.0'])
])
def test_quantiles_to_outputs(quantiles, expected_quantiles, expected_names):
    result_quantiles, result_names = quantiles_to_outputs(quantiles)
    np.testing.assert_array_equal(result_quantiles, expected_quantiles)
    np.testing.assert_array_equal(result_names, expected_names)


def test_samples_to_quantiles_df():
    #samples_to_quantiles_df unit tests
    start_date = pd.Timestamp("2023-06-01")
    end_date = pd.Timestamp("2023-06-10")
    frequency = "D"  # Daily frequency
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency).tolist()
    samples = np.random.rand(3, 200, 10)
    unique_ids = ['id1', 'id2', 'id3']
    level = np.array([10, 50, 90])
    quantiles=np.array([0.5, 0.05, 0.25, 0.45, 0.55, 0.75, 0.95])

    ret_quantiles_1, ret_df_1 = samples_to_quantiles_df(samples, unique_ids, dates, level=level)
    ret_quantiles_2, ret_df_2 = samples_to_quantiles_df(samples, unique_ids, dates, quantiles=quantiles)

    np.testing.assert_array_equal(ret_quantiles_1, quantiles)
    np.testing.assert_array_equal(ret_df_1.columns,
        ['unique_id', 'ds', 'model', 'model-median', 'model-lo-90', 'model-lo-50', 'model-lo-10', 'model-hi-10', 'model-hi-50', 'model-hi-90']
    )
    np.testing.assert_array_equal(ret_df_1["unique_id"].values,
        ['id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1',
        'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2',
        'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3']
    )
    np.testing.assert_array_equal(ret_quantiles_1, ret_quantiles_2)
    np.testing.assert_array_equal(ret_df_1["unique_id"], ret_df_2["unique_id"])
    # polars

    ret_quantiles_1, ret_df_1 = samples_to_quantiles_df(samples, unique_ids, dates, level=level, backend='polars')
    ret_quantiles_2, ret_df_2 = samples_to_quantiles_df(samples, unique_ids, dates, quantiles=quantiles, backend='polars')

    np.testing.assert_array_equal(ret_quantiles_1, quantiles)
    np.testing.assert_array_equal(ret_df_1.columns,
        ['unique_id', 'ds', 'model', 'model-median', 'model-lo-90', 'model-lo-50', 'model-lo-10', 'model-hi-10', 'model-hi-50', 'model-hi-90']
    )
    np.testing.assert_array_equal(
        list(ret_df_1["unique_id"]),
        ['id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1', 'id1',
        'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2', 'id2',
        'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3', 'id3']
    )
    np.testing.assert_array_equal(ret_quantiles_1, ret_quantiles_2)
    np.testing.assert_array_equal(ret_df_1["unique_id"], ret_df_2["unique_id"])

def test_agg_func_for_exog_vars():
    #Unit test for the aggregate function accounting for exog_vars
    df = pd.DataFrame(
        {
            'cat1': ['a', 'a', 'c'],
            'cat2': ['1', '2', '3'],
            'exog1': [4, 5, 6],
            'exog2': [7, 6, 5],
            'y': [10, 20, 30],
            'ds': ['2020-01-01', '2020-02-01', '2020-03-01']
        }
    )
    spec = [["cat1"],["cat1","cat2"]]


    Y_df_check = pd.DataFrame(
        data={
            'unique_id': ['a', 'a', 'c', 'a/1', 'a/2', 'c/3'],
            'ds': ['2020-01-01','2020-02-01','2020-03-01','2020-01-01','2020-02-01','2020-03-01'],
            'y': [10, 20, 30, 10, 20, 30],
        },
    )

    S_df_check = pd.DataFrame(
        data={
            'unique_id': ['a','c','a/1','a/2','c/3'],
            'a/1': np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            'a/2': np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            'c/3': np.array([0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        },
    )


    Y_df_check_exog = pd.DataFrame(
        data = {
            'unique_id': ['a', 'a', 'c', 'a/1', 'a/2', 'c/3'],
            'ds': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-01-01', '2020-02-01', '2020-03-01'],
            'y': [10, 20, 30, 10, 20, 30],
            'exog1_mean': [4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
            'exog2_sum': [7, 6, 5, 7, 6, 5]
        },
    )

    Y_df, S_df, tags = aggregate(
        df = df,
        spec = spec,
        exog_vars = None,
    )

    Y_df_exog, S_df_exog, tags = aggregate(
        df = df,
        spec = spec,
        exog_vars = {'exog1':'mean','exog2':'sum'},
    )

    np.testing.assert_array_equal(Y_df, Y_df_check)

    np.testing.assert_array_equal(S_df, S_df_check)

    np.testing.assert_array_equal(Y_df_exog, Y_df_check_exog)
    # polars

    df_f = pl.from_pandas(df)
    Y_df_check_f = pl.from_pandas(Y_df_check)
    S_df_check_f = pl.from_pandas(S_df_check)
    Y_df_check_exog_f = pl.from_pandas(Y_df_check_exog)

    Y_df, S_df, tags = aggregate(
        df = df_f,
        spec = spec,
        exog_vars = None,
    )

    Y_df_exog, S_df_exog, tags = aggregate(
        df = df_f,
        spec = spec,
        exog_vars = {'exog1':'mean','exog2':'sum'},
    )

    np.testing.assert_array_equal(Y_df, Y_df_check_f)

    np.testing.assert_array_equal(S_df, S_df_check_f)

    np.testing.assert_array_equal(Y_df_exog, Y_df_check_exog_f)

def test_aggregate_temporal_sparse(tourism_df, hiers_strictly):
    """aggregate_temporal with sparse_s=True should match the dense path."""
    from hierarchicalforecast.utils import SMatrix

    df = tourism_df.copy()
    df['ds'] = pd.to_datetime(df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True))

    spec_cs = [['Country'], ['Country', 'State'], ['Country', 'State', 'Region']]
    Y_df_cs, S_df_cs, tags_cs = aggregate(df, spec_cs)

    spec_te = {"year": 4, "quarter": 1}
    Y_dense, S_dense, tags_dense = aggregate_temporal(Y_df_cs, spec_te, sparse_s=False)
    Y_sparse, S_sparse, tags_sparse = aggregate_temporal(Y_df_cs, spec_te, sparse_s=True)

    assert isinstance(S_sparse, SMatrix)
    assert S_sparse._bottom_identity_verified
    np.testing.assert_array_equal(
        S_dense.values, S_sparse.to_frame(backend="pandas").values
    )


def test_smatrix_getitem():
    """Tests SMatrix.__getitem__ for id col, value col, and invalid key."""
    from scipy import sparse as sp

    from hierarchicalforecast.utils import SMatrix

    data = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    smat = SMatrix(
        sparse_matrix=sp.csc_matrix(data),
        row_labels=np.array(["top", "mid", "a", "b", "c"]),
        col_labels=np.array(["a", "b", "c"]),
    )

    # Access id column
    np.testing.assert_array_equal(smat["unique_id"], np.array(["top", "mid", "a", "b", "c"]))

    # Access a value column
    np.testing.assert_array_equal(smat["a"], np.array([1, 0, 1, 0, 0]))

    # Invalid key raises KeyError
    with pytest.raises(KeyError):
        smat["nonexistent"]


def test_smatrix_clear_cache():
    """Tests that clear_cache() releases cached representations."""
    from scipy import sparse as sp

    from hierarchicalforecast.utils import SMatrix

    data = np.eye(3, dtype=np.float64)
    smat = SMatrix(
        sparse_matrix=sp.csc_matrix(data),
        row_labels=np.array(["a", "b", "c"]),
        col_labels=np.array(["a", "b", "c"]),
    )

    # Populate caches
    smat.to_dense()
    smat.to_csr()
    smat.to_frame(backend="pandas")
    assert smat._dense is not None
    assert smat._csr is not None
    assert len(smat._frames) > 0

    smat.clear_cache()
    assert smat._dense is None
    assert smat._csr is None
    assert len(smat._frames) == 0

    # Should still work after clearing
    dense = smat.to_dense()
    np.testing.assert_array_equal(dense, np.eye(3))


def test_cov_equivalence():
    # test covariance equivalence
    n_samples = 100
    n_hiers = 10
    y_insample = np.random.rand(n_samples, n_hiers)
    y_hat_insample = np.random.rand(n_samples, n_hiers)
    residuals = (y_insample - y_hat_insample)
    nan_mask = np.isnan(residuals)

    # Check equivalence of covariance functions in case of no nans
    W_nb = _ma_cov(residuals, ~nan_mask)
    W_np = np.cov(residuals)
    np.testing.assert_allclose(W_nb, W_np, atol=1e-6)

    # Check equivalence of shrunk covariance functions in case of no nans
    W_ss_nonan = _shrunk_covariance_schaferstrimmer_no_nans(residuals, 2e-8)
    W_ss_nan = _shrunk_covariance_schaferstrimmer_with_nans(residuals, ~nan_mask, 2e-8)
    np.testing.assert_allclose(W_ss_nan, W_ss_nonan, atol=1e-6)

    # Check equivalence of diagonal elements of shrunk W to non-shrunk W in case of no nans
    np.testing.assert_allclose(np.diag(W_np), np.diag(W_ss_nan), atol=1e-6)


def _calendar_frames(backend="pandas"):
    """Fiscal calendar with a regular week, a short week and a month-crossing week.

    Business days are Mon-Fri. Week W02 has only four business days (short),
    week W03 starts in January and ends in February (crosses a natural month).
    """
    days = []
    for day in range(1, 6):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "2024-W01", "2024-M01"))
    for day in range(8, 12):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "2024-W02", "2024-M01"))
    for day in range(29, 32):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "2024-W03", "2024-M01"))
    for day in range(1, 3):
        days.append((pd.Timestamp(f"2024-02-{day:02d}"), "2024-W03", "2024-M02"))
    calendar = pd.DataFrame(days, columns=["ds", "fiscal_week", "fiscal_month"])
    if backend == "polars":
        return pl.from_pandas(calendar), days
    return calendar, days


def _calendar_data(days, missing=(), backend="pandas"):
    rows = []
    for ds, _, _ in days:
        if ds in missing:
            continue
        for unique_id in ("A", "B"):
            rows.append({"unique_id": unique_id, "ds": ds, "y": float(ds.day)})
    df = pd.DataFrame(rows)
    if backend == "polars":
        return pl.from_pandas(df)
    return df


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_aggregate_temporal_calendar_short_and_crossing_weeks(backend):
    calendar, days = _calendar_frames(backend)
    df = _calendar_data(days, backend=backend)
    spec = {"week": "fiscal_week", "month": "fiscal_month"}

    Y_df, S_df, tags, coverage = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec=spec
    )
    Y_df = Y_df.to_pandas() if backend == "polars" else Y_df
    coverage = coverage.to_pandas() if backend == "polars" else coverage

    # Every calendar group participates because all days are observed.
    assert list(tags["month"]) == ["month//2024-M01", "month//2024-M02"]
    assert list(tags["week"]) == ["week//2024-W01", "week//2024-W02", "week//2024-W03"]
    # Bottom level is made of business days, labelled from the calendar.
    assert tags["day"][0] == "day//2024-01-01"
    assert len(tags["day"]) == len(days)

    # Period identities are preserved from the calendar, not renumbered.
    assert set(tags["week"]) == {"week//2024-W01", "week//2024-W02", "week//2024-W03"}
    assert "month//2024-M02" in set(tags["month"])

    # The short week is labelled short; the month-crossing week is complete.
    week_cov = coverage[coverage["level"] == "week"].set_index("temporal_id")
    assert week_cov.loc["week//2024-W02", "is_short"]
    assert not week_cov.loc["week//2024-W01", "is_short"]
    assert week_cov.loc["week//2024-W03", "status"] == "complete"
    assert week_cov.loc["week//2024-W02", "status"] == "short"

    # Aggregated values equal the sum of their member business days.
    month_one = Y_df[(Y_df["temporal_id"] == "month//2024-M01") & (Y_df["unique_id"] == "A")]
    jan_days = [d for d, _, m in days if m == "2024-M01"]
    assert month_one["y"].iloc[0] == float(sum(d.day for d in jan_days))
    # W03 spans two natural months but is a single fiscal week.
    week_three = Y_df[(Y_df["temporal_id"] == "week//2024-W03") & (Y_df["unique_id"] == "A")]
    assert week_three["y"].iloc[0] == float(29 + 30 + 31 + 1 + 2)

    # The summing matrix aggregates each period over exactly its business days.
    S_np = S_df.to_pandas() if backend == "polars" else S_df
    S_values = S_np.set_index("temporal_id")
    row_sum = S_values.sum(axis=1).to_dict()
    assert row_sum["week//2024-W02"] == 4  # short week has four business days
    assert row_sum["week//2024-W03"] == 5  # crossing week still has five


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_aggregate_temporal_calendar_excludes_incomplete_periods(backend):
    calendar, days = _calendar_frames(backend)
    # Drop the last day of the short week for both series: the group is
    # incomplete, so neither the week nor its month may participate.
    missing = {pd.Timestamp("2024-01-11")}
    df = _calendar_data(days, missing=missing, backend=backend)
    spec = {"week": "fiscal_week", "month": "fiscal_month"}

    Y_df, S_df, tags, coverage = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec=spec
    )
    coverage = coverage.to_pandas() if backend == "polars" else coverage

    assert "week//2024-W02" not in set(tags["week"])
    # W02 belongs to M01, so M01 must also drop out; M02 stays complete.
    assert "month//2024-M01" not in set(tags["month"])
    assert list(tags["month"]) == ["month//2024-M02"]

    week_two = coverage[coverage["temporal_id"] == "week//2024-W02"].iloc[0]
    assert not week_two["is_complete"]
    assert not week_two["participates"]
    assert week_two["status"] == "incomplete"
    assert week_two["expected_days"] == 4
    assert week_two["covered_days_min"] == 3

    # The dropped day is not in the bottom level either.
    assert "day//2024-01-11" not in set(tags["day"])
    # No identifier from a dropped group leaks into Y.
    Y_df = Y_df.to_pandas() if backend == "polars" else Y_df
    assert "week//2024-W02" not in set(Y_df["temporal_id"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_aggregate_temporal_calendar_partial_series_drops_group(backend):
    calendar, days = _calendar_frames(backend)
    rows = []
    for ds, _, _ in days:
        for unique_id in ("A", "B"):
            # Series B never reports the first Tuesday of week W01.
            if unique_id == "B" and ds == pd.Timestamp("2024-01-02"):
                continue
            rows.append({"unique_id": unique_id, "ds": ds, "y": float(ds.day)})
    df = pd.DataFrame(rows)
    df = pl.from_pandas(df) if backend == "polars" else df

    _, _, tags, coverage = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec={"week": "fiscal_week", "month": "fiscal_month"}
    )
    coverage = coverage.to_pandas() if backend == "polars" else coverage

    assert "week//2024-W01" not in set(tags["week"])
    week_one = coverage[coverage["temporal_id"] == "week//2024-W01"].iloc[0]
    assert week_one["n_series_covered"] == 1
    assert week_one["covered_days_min"] == 4
    assert week_one["covered_days_max"] == 5


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_aggregate_temporal_calendar_sparse_matches_dense(backend):
    calendar, days = _calendar_frames(backend)
    df = _calendar_data(days, backend=backend)
    spec = {"week": "fiscal_week", "month": "fiscal_month"}

    Y_dense, S_dense, tags_dense, _ = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec=spec
    )
    Y_sparse, S_sparse, tags_sparse, _ = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec=spec, sparse_s=True
    )

    assert S_sparse.check_bottom_identity()
    np.testing.assert_array_equal(
        S_dense.drop(columns="temporal_id").values
        if backend == "pandas"
        else S_dense.drop("temporal_id").to_numpy(),
        S_sparse.to_dense(),
    )
    for key in tags_dense:
        np.testing.assert_array_equal(tags_dense[key], tags_sparse[key])


def test_aggregate_temporal_calendar_validation():
    calendar, days = _calendar_frames("pandas")
    df = _calendar_data(days)

    # Duplicate calendar timestamps are rejected.
    duplicated = pd.concat([calendar, calendar.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per business day"):
        aggregate_temporal_calendar(df=df, calendar=duplicated, spec={"week": "fiscal_week"})

    # Data timestamps absent from the calendar are rejected.
    extra = pd.concat(
        [df, pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-03-01")], "y": [1.0]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="absent from `calendar`"):
        aggregate_temporal_calendar(df=extra, calendar=calendar, spec={"week": "fiscal_week"})

    # A missing level column is reported.
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_temporal_calendar(df=df, calendar=calendar, spec={"week": "nope"})

    # Using the time column as a level is rejected.
    with pytest.raises(ValueError, match="cannot be used as a temporal level"):
        aggregate_temporal_calendar(df=df, calendar=calendar, spec={"year": "ds"})

    # A time format that collapses two business days onto one label is rejected.
    intraday_calendar = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 20:00"]),
            "fiscal_week": ["2024-W01", "2024-W01"],
        }
    )
    intraday_df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 20:00"]),
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate business-day labels"):
        aggregate_temporal_calendar(
            df=intraday_df, calendar=intraday_calendar, spec={"week": "fiscal_week"}
        )


def test_aggregate_temporal_calendar_regular_period_uses_modal_length():
    """A level whose first period is short is not treated as always short."""
    # Week W00 has only one business day; the following weeks have three.
    days = [(pd.Timestamp("2024-01-01"), "2024-W00", "2024-M01")]
    for start in (2, 9, 16):
        for offset in range(3):
            days.append(
                (pd.Timestamp("2024-01-01") + pd.Timedelta(days=start + offset - 1),
                 "2024-W0" + str(start // 7 + 1), "2024-M01")
            )
    calendar = pd.DataFrame(days, columns=["ds", "fiscal_week", "fiscal_month"])
    df = pd.DataFrame(
        [
            {"unique_id": unique_id, "ds": ds, "y": 1.0}
            for ds, _, _ in days
            for unique_id in ("A", "B")
        ]
    )
    _, _, tags, coverage = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec={"week": "fiscal_week"}
    )
    coverage = coverage.set_index("temporal_id")
    # One-day week is short relative to the modal three-day week.
    assert coverage.loc["week//2024-W00", "is_short"]
    # Full three-day weeks are not flagged short.
    assert not coverage.loc["week//2024-W01", "is_short"]
    assert set(tags["week"]) == {
        "week//2024-W00",
        "week//2024-W01",
        "week//2024-W02",
        "week//2024-W03",
    }


def test_aggregate_temporal_calendar_coverage_report_distinguishes_statuses():
    """A short week (fully observed) differs from a week with missing days."""
    days = []
    # W01: five business days, fully observed -> complete
    for day in range(1, 6):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "W01"))
    # W02: four business days, fully observed -> short but complete
    for day in range(8, 12):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "W02"))
    # W03: five business days, one missing -> incomplete
    for day in range(15, 20):
        days.append((pd.Timestamp(f"2024-01-{day:02d}"), "W03"))
    calendar = pd.DataFrame(days, columns=["ds", "fiscal_week"])
    rows = [
        {"unique_id": uid, "ds": ds, "y": 1.0}
        for ds, _ in days
        for uid in ("A", "B")
        if ds != pd.Timestamp("2024-01-17")  # one missing day in W03
    ]
    df = pd.DataFrame(rows)

    _, _, tags, coverage = aggregate_temporal_calendar(
        df=df, calendar=calendar, spec={"week": "fiscal_week"}
    )
    coverage = coverage.set_index("temporal_id")

    assert coverage.loc["week//W01", "status"] == "complete"
    assert coverage.loc["week//W02", "status"] == "short"
    assert coverage.loc["week//W03", "status"] == "incomplete"

    # Only fully observed weeks enter the hierarchy.
    assert set(tags["week"]) == {"week//W01", "week//W02"}
    assert "week//W03" not in set(tags["week"])
    # The short week is present and not confused with the incomplete one.
    assert coverage.loc["week//W02", "participates"]
    assert not coverage.loc["week//W03", "participates"]
