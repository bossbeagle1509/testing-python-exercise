"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np


@pytest.fixture
def solver():
    return SolveDiffusion2D()


def test_initialize_physical_parameters(solver):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver.initialize_domain(w=2.0, h=4.0, dx=0.1, dy=0.1)

    assert solver.nx == 20
    assert solver.ny == 40
    assert solver.dx == 0.1
    assert solver.dy == 0.1
    assert solver.w == 2.0
    assert solver.h == 4.0

    solver.initialize_physical_parameters(d=4.0, T_cold=300.0, T_hot=700.0)

    assert solver.D == 4.0
    assert solver.T_cold == 300.0
    assert solver.T_hot == 700.0

    expected_dt = 0.000625

    assert pytest.approx(solver.dt) == expected_dt


def test_set_initial_condition(solver):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver.initialize_domain(w=10.0, h=10.0, dx=0.1, dy=0.1)

    assert solver.dx == 0.1
    assert solver.dy == 0.1
    assert solver.w == 10.0
    assert solver.h == 10.0

    solver.initialize_physical_parameters(d=4.0, T_cold=300.0, T_hot=700.0)

    assert solver.D == 4.0
    assert solver.T_cold == 300.0
    assert solver.T_hot == 700.0

    expected_dt = 0.000625

    assert pytest.approx(solver.dt) == expected_dt

    u = solver.T_cold * np.ones((solver.nx, solver.ny))

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                u[i, j] = solver.T_hot

    solver_u = solver.set_initial_condition()

    assert np.allclose(solver_u, u)
    assert solver_u.shape == (100, 100)
    assert solver_u[0, 0] == 300.0
    assert solver_u[50, 50] == 700.0
    assert np.all((solver_u == 300.0) | (solver_u == 700.0))
    assert np.all(solver_u >= 300.0) and np.all(solver_u <= 700.0)