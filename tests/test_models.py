from layers import initializer


class TestInitializer:

    def test_initializer(self):
        params = initializer(2, 2)()
        assert params['W'].shape == (2, 2)
        assert params['b'].shape == (2, 1)