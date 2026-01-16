"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np
import pytest


@pytest.fixture
def solver():
    """Fixture to provide a fresh instance of the solver for each test."""
    return SolveDiffusion2D()


def test_initialize_domain(solver):
    """initialize_domain sets attributes and computes nx/ny correctly."""
    solver.initialize_domain(w=2.0, h=4.0, dx=0.1, dy=0.1)

    assert solver.nx == 20
    assert solver.ny == 40
    assert solver.dx == 0.1
    assert solver.dy == 0.1
    assert solver.w == 2.0
    assert solver.h == 4.0
    

def test_initialize_physical_parameters(solver):
    """initialize_physical_parameters stores values and computes stable dt."""
    # Manually setting dependencies to avoid calling initialize_domain
    solver.dx = 0.1
    solver.dy = 0.1

    solver.initialize_physical_parameters(d=4.0, T_cold=300.0, T_hot=700.0)
    
    expected_dt = 0.000625
    assert solver.D == 4.0
    assert solver.T_cold == 300.0
    assert solver.T_hot == 700.0

    assert pytest.approx(solver.dt) == expected_dt


def test_set_initial_condition(solver):
    """
    Checks function SolveDiffusion2D.set_initial_function
    """
    solver.nx = 100
    solver.ny = 100
    solver.dx = 0.1
    solver.dy = 0.1
    solver.T_cold = 300.0
    solver.T_hot = 700.0
    
    u = solver.set_initial_condition()
    
    # Check shape
    assert u.shape == (100, 100)
    
    # Check a corner (should be cold)
    assert u[0, 0] == 300.0
    
    # Check the center (5, 5) -> index (50, 50) (should be hot)
    assert u[50, 50] == 700.0
    
    # Check that we actually have both temperatures present
    assert np.all((u == 300.0) | (u == 700.0))