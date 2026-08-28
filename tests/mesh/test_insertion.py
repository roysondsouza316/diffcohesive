"""2-element cohesive insertion test: a unit square split into two
triangles along the diagonal; insert a cohesive interface along that diagonal."""

import torch

from diffcohesive.mesh import insert_cohesive_interface


def _two_triangle_mesh():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return points, elements


def test_diagonal_split_duplicates_two_nodes():
    points, elements = _two_triangle_mesh()
    result = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])

    assert result.points.shape[0] == points.shape[0] + 2
    # Triangle A (index 0, lowest-indexed on the diagonal) keeps its original node ids.
    assert torch.equal(result.elements[0], torch.tensor([0, 1, 2]))
    # Triangle B must have had nodes 0 and 2 replaced by fresh duplicate ids (>= 4), node 3 untouched.
    row_b = result.elements[1].tolist()
    assert row_b[2] == 3
    assert row_b[0] >= 4 and row_b[1] >= 4
    assert row_b[0] != row_b[1]


def test_cohesive_connectivity_is_zero_thickness():
    points, elements = _two_triangle_mesh()
    result = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])

    assert result.cohesive_connectivity.shape == (1, 4)
    a0, b0, b1, a1 = result.cohesive_connectivity[0].tolist()
    # Bottom-side nodes are the original diagonal (0, 2); top-side are their duplicates.
    assert {a0, b0} == {0, 2}
    assert a1 != a0 and a1 >= 4
    assert b1 != b0 and b1 >= 4
    # Zero-thickness: duplicated nodes sit at the same physical location as their originals.
    assert torch.allclose(result.points[a0], result.points[a1])
    assert torch.allclose(result.points[b0], result.points[b1])


def test_non_crack_nodes_untouched():
    points, elements = _two_triangle_mesh()
    result = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])

    # Node 1 and node 3 are not on the crack path and must appear unduplicated in both
    # elements that reference them.
    assert 1 in result.elements[0].tolist()
    assert 3 in result.elements[1].tolist()
    assert result.points.shape[0] == 6
    assert torch.allclose(result.points[4], points[0])
    assert torch.allclose(result.points[5], points[2])
