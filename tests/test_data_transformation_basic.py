import pandas as pd
from tools import sort_dataframe, group_and_aggregate, create_pivot_table


def test_sort_dataframe():
    df = pd.DataFrame({'a': [3,1,2]})
    out = sort_dataframe(df, by='a', ascending=True)
    assert out['a'].tolist() == [1,2,3]


def test_group_and_aggregate_mean():
    df = pd.DataFrame({'grp': ['x','x','y'], 'val': [1,2,3]})
    out = group_and_aggregate(df, group_by=['grp'], agg_dict={'val': 'mean'})
    assert set(out['grp']) == {'x','y'}
    assert 'val' in out.columns


def test_create_pivot_table():
    df = pd.DataFrame({
        'idx': ['A','A','B','B'],
        'col': ['M','N','M','N'],
        'val': [10, 20, 30, 40]
    })
    out = create_pivot_table(df, index='idx', columns='col', values='val', aggfunc='mean')
    # reset_index applied, so 'idx' is a column
    assert 'idx' in out.columns
