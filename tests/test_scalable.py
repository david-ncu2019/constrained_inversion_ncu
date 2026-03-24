import numpy as np
import pytest

from src.config import SyntheticDataset, SystemConfig
from src.scalable_solvers import solve_bayesian_scalable
from src.solvers import solve_bayesian_depth_prior
from src.synthetic import generate_synthetic_world


@pytest.fixture
def dataset() -> SyntheticDataset:
    # Use a small grid so the dense version doesn't OOM
    config = SystemConfig.create_scalable(
        grid_size=10, n_layers=5, num_wells=5, seed=123, sigma_insar=0.1, sigma_well=0.01, ell=3.0, ell_z=0.5
    )
    return generate_synthetic_world(config, seed=456)


def test_scalable_bayesian_matches_dense(dataset: SyntheticDataset) -> None:
    """
    Test that the scalable (memory-efficient) Bayesian solver yields the exact
    same posterior mean as the dense mathematical formulation for the same config.
    """
    m_post_dense, _ = solve_bayesian_depth_prior(dataset)
    m_post_scalable, var_post_scalable = solve_bayesian_scalable(dataset, chunk_size=100)

    # Check shape
    assert m_post_scalable.shape == (500,)
    assert var_post_scalable.shape == (500,)

    # Check that the scalable solution matches the dense matrix formulation
    np.testing.assert_allclose(m_post_scalable, m_post_dense, atol=1e-5)
