"""Mesh-level friction validation (Alfano-Sacco model wired through the full assembly/solver
stack, exercising the generalized state_dim=2 history plumbing): two stacked quads with a
frictional cohesive interface, pressed together and sheared past complete decohesion. The
residual shear reaction must equal mu times the normal reaction (Coulomb), and the interface
must shed all *cohesive* shear resistance (damage = 1) while retaining frictional resistance."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import FrictionalCohesiveTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve


def _sandwich_model(mu):
    # Two stacked unit quads, horizontal interface at y=1 (edge (2,3)).
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 3, 2], [2, 3, 5, 4]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(2, 3)])
    law = FrictionalCohesiveTSL(
        sigma0=3.0, tau0=3.0, G_c1=0.1, G_c2=0.1, K1=1.0e4, K2=1.0e4, mu=mu
    )
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"quad": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1.0e6,  # stiff bulk so the prescribed displacements land almost entirely on the interface
        nu=0.0,
    )
    return model, ins


def test_sheared_compressed_interface_leaves_coulomb_residual():
    mu = 0.5
    model, ins = _sandwich_model(mu)
    dtype = model.points.dtype

    bottom_nodes = torch.tensor([0, 1])
    top_nodes = torch.unique(ins.elements[1])  # the upper quad's four nodes
    bottom_dofs = model.dof_indices(bottom_nodes)
    top_dofs = model.dof_indices(top_nodes).reshape(-1, 2)
    top_x, top_y = top_dofs[:, 0], top_dofs[:, 1]

    compress = -1.0e-3  # prescribed downward displacement of the whole upper block
    sc2 = 2.0 * 0.1 / 3.0
    kappa = model.init_history()
    assert kappa.shape == (model.n_coh, model.n_quad, 2)  # state_dim=2 plumbing

    u = torch.zeros(model.n_dof, dtype=dtype)
    shear_reactions, normal_reactions, damages = [], [], []
    for s in torch.linspace(0.0, 3.0 * sc2, 60, dtype=dtype):
        prescribed_dofs = torch.cat([bottom_dofs, top_x, top_y])
        prescribed_values = torch.cat(
            [
                torch.zeros(bottom_dofs.numel(), dtype=dtype),
                torch.full((top_x.numel(),), s.item(), dtype=dtype),
                torch.full((top_y.numel(),), compress, dtype=dtype),
            ]
        )
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u, max_iter=80)
        assert result.converged
        u, kappa = result.u, result.kappa
        n_top = top_x.numel()
        shear_reactions.append(result.reaction[bottom_dofs.numel():bottom_dofs.numel() + n_top].sum().item())
        normal_reactions.append(result.reaction[bottom_dofs.numel() + n_top:].sum().item())
        damages.append(result.damage.max().item())

    assert damages[-1] > 0.999  # complete decohesion
    N = abs(normal_reactions[-1])
    T = abs(shear_reactions[-1])
    assert N > 0.0
    # Coulomb: residual shear transmitted across the fully damaged interface = mu * N.
    assert abs(T - mu * N) / (mu * N) < 5e-2

    # Control experiment: with mu = 0 the residual shear vanishes.
    model0, ins0 = _sandwich_model(0.0)
    kappa0 = model0.init_history()
    u0 = torch.zeros(model0.n_dof, dtype=dtype)
    for s in torch.linspace(0.0, 3.0 * sc2, 60, dtype=dtype):
        prescribed_values = torch.cat(
            [
                torch.zeros(bottom_dofs.numel(), dtype=dtype),
                torch.full((top_x.numel(),), s.item(), dtype=dtype),
                torch.full((top_y.numel(),), compress, dtype=dtype),
            ]
        )
        result0 = newton_solve(model0, prescribed_dofs, prescribed_values, kappa0, u0=u0, max_iter=80)
        assert result0.converged
        u0, kappa0 = result0.u, result0.kappa
    T0 = abs(result0.reaction[bottom_dofs.numel():bottom_dofs.numel() + top_x.numel()].sum().item())
    assert T0 < 0.05 * T
