import numpy as np
import pytest

from src.config import SyntheticDataset, SystemConfig
from src.solvers import (
    solve_bayesian,
    solve_bayesian_anisotropic,
    solve_bayesian_depth_prior,
    solve_tikhonov,
    solve_tikhonov_nnls,
)
from src.synthetic import generate_synthetic_world


@pytest.fixture
def dataset() -> SyntheticDataset:
    config = SystemConfig(sigma_insar=0.1, sigma_well=0.01)
    return generate_synthetic_world(config, seed=123)


def test_solve_tikhonov(dataset: SyntheticDataset) -> None:
    m_hat = solve_tikhonov(dataset)

    assert m_hat.shape == (500,)

    # Check that it's a reasonable approximation (MSE shouldn't be insanely high)
    mse = np.mean((m_hat - dataset.m_true) ** 2)
    assert mse < 10.0  # Just a sanity bound


def test_solve_bayesian(dataset: SyntheticDataset) -> None:
    m_post, C_post = solve_bayesian(dataset)

    assert m_post.shape == (500,)
    assert C_post.shape == (500, 500)

    # Check that it's a reasonable approximation
    mse = np.mean((m_post - dataset.m_true) ** 2)
    assert mse < 10.0

    # Covariance should be symmetric
    np.testing.assert_allclose(C_post, C_post.T, atol=1e-5)

    # Diagonal should be positive (variances)
    assert np.all(np.diag(C_post) > 0)


def test_solve_tikhonov_nnls(dataset: SyntheticDataset) -> None:
    m_hat = solve_tikhonov_nnls(dataset)

    assert m_hat.shape == (500,)
    assert np.all(m_hat >= 0), "NNLS solution must be non-negative"

    mse = np.mean((m_hat - dataset.m_true) ** 2)
    assert mse < 10.0


def test_solve_bayesian_anisotropic(dataset: SyntheticDataset) -> None:
    m_post, C_post = solve_bayesian_anisotropic(dataset)

    assert m_post.shape == (500,)
    assert C_post.shape == (500, 500)

    mse = np.mean((m_post - dataset.m_true) ** 2)
    assert mse < 10.0

    np.testing.assert_allclose(C_post, C_post.T, atol=1e-5)
    assert np.all(np.diag(C_post) > 0)


def test_solve_bayesian_depth_prior(dataset: SyntheticDataset) -> None:
    m_post, C_post = solve_bayesian_depth_prior(dataset)

    assert m_post.shape == (500,)
    assert C_post.shape == (500, 500)

    mse = np.mean((m_post - dataset.m_true) ** 2)
    assert mse < 10.0

    np.testing.assert_allclose(C_post, C_post.T, atol=1e-5)
    assert np.all(np.diag(C_post) > 0)
