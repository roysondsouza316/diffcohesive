"""Structured two-arm (double-cantilever-beam-like) specimen mesh builder for mesh-level
validation (DCB, and the mixed-mode beam of examples/mixed_mode_beam.py).

Builds a rectangular specimen as a structured triangle grid (each quad cell split by one
diagonal), with the *entire* mid-thickness row of nodes pre-split via
``insert_cohesive_interface`` so the two arms carry independent DOFs along the whole length.
The caller selects, via ``crack_length``, how many of the returned mid-line node-pairs get a
cohesive element (the "bonded ligament") vs. none at all (the "pre-crack", left traction-free
since neither a bulk nor a cohesive element bridges those duplicated nodes).
"""

from dataclasses import dataclass

import torch

from .insertion import insert_cohesive_interface, insert_cohesive_interface_3d


@dataclass
class SpecimenMesh:
    points: torch.Tensor                  # (n_points, 2)
    elements: torch.Tensor                # (n_elem, 3|4) bulk connectivity (tri or quad)
    cohesive_connectivity: torch.Tensor   # (n_bonded, 4) bonded-ligament rows only, x ascending
    length: float
    arm_height: float
    crack_length: float
    dx: float
    dy: float
    tip_bottom: int                       # node id at (x=0, y=0)
    tip_top: int                          # node id at (x=0, y=2*arm_height)
    right_edge_nodes: torch.Tensor        # node ids with x == length (includes both mid-line
                                           # duplicates), for clamping the far end
    cell_type: str = "triangle"           # meshio cell name for CohesiveMeshModel


@dataclass
class PulloutMesh:
    """Half-model of the mode-II pull-out test of Alfano, CST 66 (2006) §3.4: a central lamina
    (full thickness 2*lamina_half_thickness) bonded over ``bond_length`` between two blocks and
    pulled out axially. Geometry interpretation (the paper gives the mesh and load level rather
    than a dimensioned drawing): lamina runs the full specimen height; the block spans the
    lower ``bond_length``; symmetry on the lamina mid-plane. The half-model peak transferable
    load is then ~ tau0 * bond_length * width, i.e. ~3000 N for the paper's properties -- twice
    that, 6000 N, for the full model, matching the scale of the paper's Fig. 10."""
    points: torch.Tensor
    elements: torch.Tensor                # (n_elem, 4) quad connectivity
    cohesive_connectivity: torch.Tensor   # one row per bonded interface edge, y ascending
    sym_nodes: torch.Tensor               # lamina mid-plane nodes (u_x = 0)
    fixed_nodes: torch.Tensor             # block bottom-edge nodes (fully fixed)
    load_nodes: torch.Tensor              # lamina top-edge nodes (prescribed u_y)
    cell_type: str = "quad"


def build_double_cantilever_mesh(
    length: float,
    arm_height: float,
    crack_length: float,
    nx: int,
    ny: int,
    element_type: str = "triangle",
    dtype: torch.dtype = torch.float64,
) -> SpecimenMesh:
    """A rectangular specimen of total thickness ``2*arm_height`` and length ``length``, split
    along its mid-thickness line. ``nx`` cells span the length, ``ny`` cells span *each* arm
    (so ``2*ny`` cells through the full thickness). ``crack_length`` is measured from the x=0
    end (where the opening displacement is applied) and is rounded to the nearest cell boundary.
    ``element_type``: "triangle" (each grid cell split by a diagonal) or "quad" (one bilinear
    quadrilateral per grid cell -- markedly less bending-stiff than linear triangles when only
    a few elements span each arm's thickness, matching the quad meshes used in the CZM
    validation literature, e.g. Alfano CST 2006).
    """
    if element_type not in ("triangle", "quad"):
        raise ValueError(f"element_type must be 'triangle' or 'quad', got {element_type!r}")
    n_cols = nx + 1
    n_rows = 2 * ny + 1
    dx = length / nx
    dy = arm_height / ny

    def node_id(row: int, col: int) -> int:
        return row * n_cols + col

    xs = torch.arange(n_cols, dtype=dtype) * dx
    ys = torch.arange(n_rows, dtype=dtype) * dy
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    elements = []
    for r in range(2 * ny):
        for c in range(nx):
            p00 = node_id(r, c)
            p10 = node_id(r, c + 1)
            p01 = node_id(r + 1, c)
            p11 = node_id(r + 1, c + 1)
            if element_type == "triangle":
                elements.append([p00, p10, p11])
                elements.append([p00, p11, p01])
            else:
                elements.append([p00, p10, p11, p01])
    elements = torch.tensor(elements, dtype=torch.long)

    r_mid = ny
    crack_edges = [(node_id(r_mid, c), node_id(r_mid, c + 1)) for c in range(nx)]
    ins = insert_cohesive_interface(points, elements, crack_edges=crack_edges)

    n_precrack_cells = int(round(crack_length / dx))
    cohesive_connectivity = ins.cohesive_connectivity[n_precrack_cells:]

    tip_bottom = node_id(0, 0)
    tip_top = node_id(2 * ny, 0)
    # all points whose x-coordinate equals `length` (within fp tolerance) -- this naturally
    # includes both mid-line duplicate nodes at the far end, not just one.
    is_right = (ins.points[:, 0] - length).abs() < 1e-9 * max(length, 1.0)
    right_edge_nodes = torch.nonzero(is_right, as_tuple=True)[0]

    return SpecimenMesh(
        points=ins.points,
        elements=ins.elements,
        cohesive_connectivity=cohesive_connectivity,
        length=length,
        arm_height=arm_height,
        crack_length=n_precrack_cells * dx,
        dx=dx,
        dy=dy,
        tip_bottom=tip_bottom,
        tip_top=tip_top,
        right_edge_nodes=right_edge_nodes,
        cell_type="triangle" if element_type == "triangle" else "quad",
    )


@dataclass
class SpecimenMesh3D:
    points: torch.Tensor                  # (n_points, 3)
    elements: torch.Tensor                # (n_hex, 8) hexahedron connectivity
    cohesive_connectivity: torch.Tensor   # (n_bonded, 8): [b0..b3, t0..t3], t_i pairs b_i
    length: float
    arm_height: float
    width: float
    crack_length: float
    tip_bottom_nodes: torch.Tensor        # node line x=0, y=0 (all z)
    tip_top_nodes: torch.Tensor           # node line x=0, y=2*arm_height (all z)
    right_edge_nodes: torch.Tensor        # all nodes with x == length
    cell_type: str = "hexahedron"


def build_double_cantilever_mesh_3d(
    length: float,
    arm_height: float,
    width: float,
    crack_length: float,
    nx: int,
    ny: int,
    nz: int,
    dtype: torch.dtype = torch.float64,
) -> SpecimenMesh3D:
    """3D counterpart of :func:`build_double_cantilever_mesh`: a structured hexahedral DCB
    specimen (two arms of thickness ``arm_height``, out-of-plane ``width``), split along the
    whole mid-thickness plane via :func:`insert_cohesive_interface_3d`; only the bonded
    ligament (x >= crack_length) receives cohesive elements."""
    n_cols, n_rows, n_lay = nx + 1, 2 * ny + 1, nz + 1
    dx, dy, dz = length / nx, arm_height / ny, width / nz

    def gid(ix: int, iy: int, iz: int) -> int:
        return (iz * n_rows + iy) * n_cols + ix

    pts = torch.tensor(
        [[ix * dx, iy * dy, iz * dz]
         for iz in range(n_lay) for iy in range(n_rows) for ix in range(n_cols)],
        dtype=dtype,
    )

    hexes = []
    for iz in range(nz):
        for iy in range(2 * ny):
            for ix in range(nx):
                # meshio hexahedron: bottom face (z-) counterclockwise, then top face (z+).
                # Build with the *y*-direction as the through-thickness axis of the DCB, so
                # "bottom/top" here are just the two z-layers of the cell.
                hexes.append([
                    gid(ix, iy, iz), gid(ix + 1, iy, iz), gid(ix + 1, iy + 1, iz), gid(ix, iy + 1, iz),
                    gid(ix, iy, iz + 1), gid(ix + 1, iy, iz + 1), gid(ix + 1, iy + 1, iz + 1), gid(ix, iy + 1, iz + 1),
                ])
    hexes = torch.tensor(hexes, dtype=torch.long)

    # Mid-plane crack faces at y = arm_height: the quad (in cyclic order) between cell rows.
    r_mid = ny
    crack_faces = []
    for iz in range(nz):
        for ix in range(nx):
            crack_faces.append((
                gid(ix, r_mid, iz), gid(ix + 1, r_mid, iz),
                gid(ix + 1, r_mid, iz + 1), gid(ix, r_mid, iz + 1),
            ))

    ins = insert_cohesive_interface_3d(pts, hexes, crack_faces=crack_faces)

    # Bonded ligament: faces whose centroid x >= crack_length (rounded to cell boundaries).
    n_precrack_cells = int(round(crack_length / dx))
    face_x = ins.points[ins.cohesive_connectivity[:, :4]].mean(dim=1)[:, 0]
    bonded = face_x > n_precrack_cells * dx - 1e-9
    cohesive_connectivity = ins.cohesive_connectivity[bonded]

    tol = 1e-9 * max(length, 1.0)
    p = ins.points
    tip_bottom_nodes = torch.nonzero((p[:, 0].abs() < tol) & (p[:, 1].abs() < tol), as_tuple=True)[0]
    tip_top_nodes = torch.nonzero(
        (p[:, 0].abs() < tol) & ((p[:, 1] - 2 * arm_height).abs() < tol), as_tuple=True)[0]
    right_edge_nodes = torch.nonzero((p[:, 0] - length).abs() < tol, as_tuple=True)[0]

    return SpecimenMesh3D(
        points=ins.points,
        elements=ins.elements,
        cohesive_connectivity=cohesive_connectivity,
        length=length,
        arm_height=arm_height,
        width=width,
        crack_length=n_precrack_cells * dx,
        tip_bottom_nodes=tip_bottom_nodes,
        tip_top_nodes=tip_top_nodes,
        right_edge_nodes=right_edge_nodes,
    )


def build_pullout_mesh(
    lamina_half_thickness: float = 2.0,
    bond_length: float = 50.0,
    free_length: float = 50.0,
    block_width: float = 50.0,
    n_lamina_cols: int = 2,
    dy: float = 1.0,
    block_dx: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> PulloutMesh:
    """Structured quad mesh for the half pull-out model (see PulloutMesh docstring). The
    lamina occupies x in [0, t], y in [0, bond+free]; the block occupies x in [t, t+block_width],
    y in [0, bond]. The bonded interface is the vertical line x = t, y in [0, bond]."""
    t = lamina_half_thickness
    height = bond_length + free_length
    n_rows = int(round(height / dy))
    n_bond_rows = int(round(bond_length / dy))
    n_block_cols = int(round(block_width / block_dx))

    xs = [i * t / n_lamina_cols for i in range(n_lamina_cols + 1)]
    xs += [t + (i + 1) * block_dx for i in range(n_block_cols)]
    ys = [r * dy for r in range(n_rows + 1)]
    n_cols_total = len(xs) - 1

    def gid(r: int, c: int) -> int:
        return r * (n_cols_total + 1) + c

    cells = []
    is_lamina_cell = []
    for r in range(n_rows):
        for c in range(n_cols_total):
            lamina = c < n_lamina_cols
            if not lamina and r >= n_bond_rows:
                continue  # block exists only over the bonded length
            cells.append([gid(r, c), gid(r, c + 1), gid(r + 1, c + 1), gid(r + 1, c)])
            is_lamina_cell.append(lamina)

    # Compact away grid nodes not referenced by any cell (the empty region above the block).
    cells_t = torch.tensor(cells, dtype=torch.long)
    used = torch.unique(cells_t)
    remap = torch.full((int(used.max()) + 1,), -1, dtype=torch.long)
    remap[used] = torch.arange(used.numel())
    cells_t = remap[cells_t]
    grid_points = torch.tensor(
        [[xs[c], ys[r]] for r in range(n_rows + 1) for c in range(n_cols_total + 1)], dtype=dtype
    )
    points = grid_points[used]

    interface_col = n_lamina_cols
    crack_edges = [
        (int(remap[gid(r, interface_col)]), int(remap[gid(r + 1, interface_col)]))
        for r in range(n_bond_rows)
    ]

    ins = insert_cohesive_interface(points, cells_t, crack_edges=crack_edges)

    is_lamina_cell = torch.tensor(is_lamina_cell, dtype=torch.bool)
    lamina_nodes = torch.unique(ins.elements[is_lamina_cell])
    block_nodes = torch.unique(ins.elements[~is_lamina_cell])
    tol = 1e-9 * max(height, 1.0)
    p = ins.points
    sym_nodes = lamina_nodes[p[lamina_nodes, 0].abs() < tol]
    load_nodes = lamina_nodes[(p[lamina_nodes, 1] - height).abs() < tol]
    fixed_nodes = block_nodes[p[block_nodes, 1].abs() < tol]

    return PulloutMesh(
        points=ins.points,
        elements=ins.elements,
        cohesive_connectivity=ins.cohesive_connectivity,
        sym_nodes=sym_nodes,
        fixed_nodes=fixed_nodes,
        load_nodes=load_nodes,
    )
