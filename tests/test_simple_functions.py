import numpy as np
import pytest

from simple_functions import factorial, my_sin, my_sum, my_torch_version


class TestSimpleFunctions(object):
    '''Class to test our simple functions are working correctly'''

    @pytest.mark.parametrize('iterable, expected', [
        ([8, 7, 5], 20),
        ((10, -2, 5, -10, 1), 4)
    ])
    def test_my_add(self, iterable, expected):
        '''Test our add function'''
        isum = my_sum(iterable)
        assert isum == expected

    @pytest.mark.parametrize('number, expected', [
        (5, 120),
        (3, 6),
        (1, 1)
    ])
    def test_factorial(self, number, expected):
        '''Test our factorial function'''
        answer = factorial(number)
        assert answer == expected

    @pytest.mark.parametrize('x_in, expected', [
        (0, 0),
        (np.pi/2, 1)
    ])
    def test_sin(self, x_in, expected):
        """ Test sin function """
        answer = my_sin(x_in)
        assert np.isclose(answer, expected)

    @pytest.mark.parametrize('expected', ['1.12.1'])
    def test_torch(self, expected):
        """ Test sin function """
        my_version = my_torch_version()
        assert my_version == expected
