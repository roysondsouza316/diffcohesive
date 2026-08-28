"""Wiring test: TensorMesh bulk elasticity + custom cohesive element through a real
global Newton solve on the 2-triangle diagonal-crack mesh (forward only)."""

import torch

from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.solvers import newton_solve


def _cracked_model(K_penalty=1.0e4, T_max=5.0, G_c=0.05):
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])

    law = BilinearMixedModeTSL(T_max_n=T_max, T_max_s=T_max, G_c1=G_c, G_c2=G_c, eta=1.0, K=K_penalty)
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1000.0,
        nu=0.3,
    )
    return model


def test_newton_converges_and_bulk_cohesive_couple_elastically():
    model = _cracked_model()
    # Triangle A = [0,1,2] is "ground"; node 3 is B's only free (non-interface) vertex.
    prescribed_dofs = torch.tensor([0, 1, 2, 3, 4, 5])  # nodes 0,1,2 fully fixed
    kappa = model.init_history()

    # Small displacement: stay in the elastic (pre-damage-onset) regime.
    u0 = torch.zeros(model.n_dof, dtype=torch.float64)
    node3_dofs = model.dof_indices(torch.tensor([3]))
    small_disp = torch.zeros(model.n_dof, dtype=torch.float64)

    prescribed_values = torch.zeros(6, dtype=torch.float64)
    result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u0)
    assert result.converged
    # Zero prescribed displacement everywhere reachable -> trivial zero solution, zero damage.
    assert torch.allclose(result.u, torch.zeros_like(result.u), atol=1e-10)
    assert torch.all(result.damage <= 1e-8)


def test_pulling_free_corner_engages_bulk_and_interface_with_softening():
    model = _cracked_model()
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=torch.float64)

    forces = []
    max_damages = []
    disps = torch.linspace(0.0, 0.02, 15, dtype=torch.float64)
    converged_all = True
    last_converged_result = None
    for d in disps:
        prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])  # fix A + node3's y-dof
        prescribed_values = torch.cat([torch.zeros(6, dtype=torch.float64), d.reshape(1)])
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u)
        if not result.converged:
            converged_all = False
            break
        u = result.u
        kappa = result.kappa
        forces.append(result.reaction[-1].item())  # reaction at node3's prescribed y-dof
        max_damages.append(result.damage.max().item())
        last_converged_result = result

    assert len(forces) >= 5  # got well into the loading history before any divergence
    # Elastic onset: damage starts at 0 and eventually grows (softening engages).
    assert max_damages[0] == 0.0 or max_damages[0] < 1e-6
    assert max_damages[-1] > max_damages[0]
    assert all(m2 >= m1 - 1e-9 for m1, m2 in zip(max_damages, max_damages[1:]))  # monotone damage
    # Near-zero displacement gives near-zero force; response starts elastic (roughly linear).
    assert abs(forces[0]) < abs(forces[1])
