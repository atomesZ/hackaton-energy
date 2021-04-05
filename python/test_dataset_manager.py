from dataset_manager import *

def test_get_X_Y():
    assert get_X_Y({}) == ([], [])
    assert get_X_Y({'A': [1, 2]}) == ([1, 2], ['A', 'A'])
    assert get_X_Y({'A': [1, 2], 'B': {3, 4}}) == ([1, 2, 3, 4], ['A', 'A', 'B', 'B'])

def test_get_X_Y_vectorized_int():
    assert get_X_Y_vectorized_int({}) == ([], [])
    assert get_X_Y_vectorized_int({'A': [1, 2]}) == ([1, 2], [[1], [1]])
    assert get_X_Y_vectorized_int({'A': [1, 2], 'B': [3, 4]}) == ([1, 2, 3, 4], [[1, 0], [1, 0], [0, 1], [0, 1]])

def test_shuffle_X_Y():
    assert np.array_equal(shuffle_X_Y([], []), (np.array([]), np.array([])))

