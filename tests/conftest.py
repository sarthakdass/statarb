import pytest

from statarb.data import simulate_prices


@pytest.fixture(scope="session")
def market():
    return simulate_prices(n_days=900, n_pairs=2, n_independent=3, seed=11)
