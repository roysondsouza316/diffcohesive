"""Quickstart: a differentiable DCB in ~20 lines.

Needs nothing but the installed package -- run from anywhere after `pip install diffcohesive`:

    python quickstart_dcb.py

Builds a small double-cantilever-beam specimen with cohesive elements on the bonded ligament,
opens the arms under displacement control (a damaging Newton solve), then repeats the solve
differentiably and returns d||u||/d(theta) for the six cohesive-law parameters via the adjoint.
"""

import torch

from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.solvers import newton_solve
from diffcohesive.diff import solve_diff, theta_from_law

# 1. FE model: DCB with a 5 mm pre-crack, cohesive elements on the bonded ligament
mesh = build_double_cantilever_mesh(length=15.0, arm_height=1.0,
                                    crack_length=5.0, nx=30, ny=4)
law = BilinearMixedModeTSL(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, K=1e4)
model = CohesiveMeshModel(mesh.points, {mesh.cell_type: mesh.elements},
                          mesh.cohesive_connectivity, law, E=1000.0, nu=0.3)

# 2. boundary conditions: clamp the far edge, open the arm tips by +/- d/2
right = model.dof_indices(mesh.right_edge_nodes)
tip_t = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
tip_b = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]
d = 0.3
dofs = torch.cat([right, tip_t.reshape(1), tip_b.reshape(1)])
vals = torch.cat([torch.zeros(right.numel(), dtype=torch.float64),
                  torch.tensor([d / 2, -d / 2], dtype=torch.float64)])

# 3a. plain forward solve...
result = newton_solve(model, dofs, vals, model.init_history())
print("reaction:", result.reaction[-2].item(), "max damage:", result.damage.max().item())

# 3b. ...or a DIFFERENTIABLE solve: d(anything)/d(law parameters) via the adjoint
theta = theta_from_law(law).requires_grad_(True)
u_star = solve_diff(theta, model, dofs, vals, model.init_history())
u_star.norm().backward()
print("d||u||/d(T_max, G_c, ...):", theta.grad)
