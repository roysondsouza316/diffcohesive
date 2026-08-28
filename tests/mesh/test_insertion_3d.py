"""3D cohesive insertion topology checks: node duplication across the crack plane, coincident
top/bottom pairs in matching order, opening-positive orientation of every cohesive face, and
the bonded-ligament subset logic of the 3D DCB builder."""

import torch

from diffcohesive.mesh import build_double_cantilever_mesh_3d, insert_cohesive_interface_3d


def test_two_hex_stack_splits_into_one_cohesive_face():
    # Two stacked unit hexes sharing a face; crack that face.
    pts = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0, 2.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=torch.float64,
    )

    def gid(ix, iy, iz):
        return (iz * 2 + iy) * 2 + ix

    hexes = torch.tensor(
        [
            [gid(0, 0, 0), gid(1, 0, 0), gid(1, 1, 0), gid(0, 1, 0),
             gid(0, 0, 1), gid(1, 0, 1), gid(1, 1, 1), gid(0, 1, 1)],
            [gid(0, 0, 1), gid(1, 0, 1), gid(1, 1, 1), gid(0, 1, 1),
             gid(0, 0, 2), gid(1, 0, 2), gid(1, 1, 2), gid(0, 1, 2)],
        ],
        dtype=torch.long,
    )
    face = (gid(0, 0, 1), gid(1, 0, 1), gid(1, 1, 1), gid(0, 1, 1))
    ins = insert_cohesive_interface_3d(pts, hexes, crack_faces=[face])

    assert ins.points.shape[0] == 12 + 4  # the 4 shared nodes duplicated once
    assert ins.cohesive_connectivity.shape == (1, 8)
    row = ins.cohesive_connectivity[0]
    # top nodes coincide with their bottom partners (zero thickness)...
    assert torch.allclose(ins.points[row[:4]], ins.points[row[4:]])
    # ...and are actually different DOF carriers.
    assert not torch.equal(row[:4], row[4:])
    # Orientation: face normal from the bottom cyclic order must point from the lower element
    # toward the upper one (+z here), so positive normal separation means opening.
    p = ins.points
    v1 = p[row[1]] - p[row[0]]
    v2 = p[row[3]] - p[row[0]]
    normal = torch.linalg.cross(v1, v2)
    assert normal[2] > 0


def test_3d_dcb_builder_bonds_only_the_ligament():
    mesh = build_double_cantilever_mesh_3d(
        length=60.0, arm_height=2.0, width=4.0, crack_length=20.0, nx=30, ny=2, nz=2
    )
    # 30 x 2 mid-plane faces total; 20 pre-crack cells x 2 layers unbonded.
    assert mesh.cohesive_connectivity.shape == (20 * 2, 8)
    face_x = mesh.points[mesh.cohesive_connectivity[:, :4]].mean(dim=1)[:, 0]
    assert face_x.min() > 20.0
    # Zero thickness everywhere.
    assert torch.allclose(
        mesh.points[mesh.cohesive_connectivity[:, :4]],
        mesh.points[mesh.cohesive_connectivity[:, 4:]],
    )
    assert mesh.tip_top_nodes.numel() == 3 and mesh.tip_bottom_nodes.numel() == 3
