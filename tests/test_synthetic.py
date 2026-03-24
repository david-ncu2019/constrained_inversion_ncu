import numpy as np
import pytest

from src.config import SystemConfig
from src.synthetic import generate_synthetic_world
from src.system import build_G_matrix, build_I_wells


@pytest.fixture
def config() -> SystemConfig:
    # Use zero noise for consistency test
    return SystemConfig(sigma_insar=0.0, sigma_well=0.0)


def test_generate_synthetic_world_shapes(config: SystemConfig) -> None:
    dataset = generate_synthetic_world(config)

    assert dataset.m_true.shape == (500,)
    assert dataset.d_insar.shape == (100,)
    assert dataset.w.shape == (25,)


def test_forward_model_consistency(config: SystemConfig) -> None:
    """Test that the observations perfectly match the forward model when noise is zero."""
    dataset = generate_synthetic_world(config)

    G = build_G_matrix(config)
    I_w = build_I_wells(config)

    # Forward project the true model
    d_insar_pred = G @ dataset.m_true
    w_pred = I_w @ dataset.m_true

    # With zero noise, they should be exactly equal
    np.testing.assert_allclose(d_insar_pred, dataset.d_insar)
    np.testing.assert_allclose(w_pred, dataset.w)


def test_synthetic_data_has_noise() -> None:
    """Test that non-zero noise actually perturbs the observations."""
    config = SystemConfig(sigma_insar=0.5, sigma_well=0.1)
    dataset = generate_synthetic_world(config)

    G = build_G_matrix(config)
    I_w = build_I_wells(config)

    d_insar_perfect = G @ dataset.m_true
    w_perfect = I_w @ dataset.m_true

    # Should not be exactly equal due to noise
    assert not np.allclose(d_insar_perfect, dataset.d_insar)
    assert not np.allclose(w_perfect, dataset.w)
