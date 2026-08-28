"""Cohesive-interface insertion: node duplication + zero-thickness element connectivity.
TensorMesh has no notion of duplicated/independent per-element DOFs, so this is a mesh-level preprocessing step: nodes along
the predefined crack path are duplicated so the two sides of the interface carry independent
DOFs, and TensorMesh's ordinary node-indexed assembly is otherwise untouched.

Supports any single-type 2D element whose nodes are ordered cyclically around the element
boundary -- triangles (M, 3) and quadrilaterals (M, 4) both qualify; the crack path is a set
of element edges.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List

import torch


@dataclass
class CohesiveInsertionResult:
    points: torch.Tensor          # (N_new, dim)
    elements: torch.Tensor        # (M, n_node) bulk connectivity (tri or quad), updated in
                                   # place for the duplicated side of the crack
    cohesive_connectivity: torch.Tensor  # (n_coh, 4): [a_bottom, b_bottom, b_top, a_top]


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def insert_cohesive_interface(
    points: torch.Tensor,
    elements: torch.Tensor,
    crack_edges: Iterable[Tuple[int, int]],
) -> CohesiveInsertionResult:
    """Duplicate nodes along ``crack_edges`` and build cohesive element connectivity.

    For each node on the crack path, elements incident to that node are partitioned into
    connected groups using only non-crack-path adjacency (i.e. two elements are adjacent iff
    they share an edge that is *not* on the crack path). The group containing the
    lowest-indexed element keeps the original node index; every other group gets an
    independent duplicate node at the same coordinates. This is exactly node duplication
    along the interface, generalized to handle nodes shared by multiple crack
    segments.
    """
    elements = elements.clone()
    n_elem = elements.shape[0]
    crack_edge_set = {_edge_key(a, b) for a, b in crack_edges}
    crack_nodes = sorted({n for e in crack_edge_set for n in e})

    # element -> its boundary edges (cyclic node ordering assumed, so this works for both
    # 3-node triangles and 4-node quadrilaterals), and edge -> elements touching it (for
    # adjacency + cohesive pairing).
    elem_edges: List[List[Tuple[int, int]]] = []
    edge_to_elems: Dict[Tuple[int, int], List[int]] = {}
    for ei in range(n_elem):
        nodes = elements[ei].tolist()
        n_node = len(nodes)
        edges = [_edge_key(nodes[i], nodes[(i + 1) % n_node]) for i in range(n_node)]
        elem_edges.append(edges)
        for e in edges:
            edge_to_elems.setdefault(e, []).append(ei)

    new_points_list = [points]
    next_point_idx = points.shape[0]

    # node_side_map[node][element_index] = the (possibly duplicated) node id that element uses.
    node_side_map: Dict[int, Dict[int, int]] = {}

    for node in crack_nodes:
        incident_elems = [ei for ei in range(n_elem) if node in elements[ei].tolist()]

        adjacency = {ei: set() for ei in incident_elems}
        for ei in incident_elems:
            for e in elem_edges[ei]:
                if e in crack_edge_set:
                    continue
                for ej in edge_to_elems[e]:
                    if ej != ei and ej in adjacency:
                        adjacency[ei].add(ej)

        visited = set()
        groups: List[List[int]] = []
        for ei in incident_elems:
            if ei in visited:
                continue
            stack = [ei]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                stack.extend(adjacency[cur] - visited)
            groups.append(sorted(comp))
        groups.sort(key=lambda g: g[0])

        mapping = {}
        for gi, group in enumerate(groups):
            if gi == 0:
                node_id = node
            else:
                node_id = next_point_idx
                next_point_idx += 1
                new_points_list.append(points[node : node + 1])
            for ei in group:
                mapping[ei] = node_id
        node_side_map[node] = mapping

    for node, mapping in node_side_map.items():
        for ei, node_id in mapping.items():
            row = elements[ei].tolist()
            row = [node_id if v == node else v for v in row]
            elements[ei] = torch.tensor(row, dtype=elements.dtype)

    cohesive_rows = []
    for a, b in sorted(crack_edge_set):
        adj = edge_to_elems[(a, b)]
        if len(adj) != 2:
            continue  # boundary edge on the crack path: no interior interface to insert
        e0, e1 = adj
        a0, b0 = node_side_map[a][e0], node_side_map[b][e0]
        a1, b1 = node_side_map[a][e1], node_side_map[b][e1]
        cohesive_rows.append([a0, b0, b1, a1])

    new_points = torch.cat(new_points_list, dim=0)
    cohesive_connectivity = torch.tensor(cohesive_rows, dtype=torch.long)
    return CohesiveInsertionResult(points=new_points, elements=elements, cohesive_connectivity=cohesive_connectivity)


# Face index pattern of the 8-node meshio "hexahedron" (bottom 0-3 counterclockwise, top 4-7),
# each face in cyclic node order.
_HEX_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]


def _face_key(nodes) -> Tuple[int, ...]:
    return tuple(sorted(nodes))


def insert_cohesive_interface_3d(
    points: torch.Tensor,
    elements: torch.Tensor,
    crack_faces: Iterable[Tuple[int, int, int, int]],
) -> CohesiveInsertionResult:
    """3D counterpart of :func:`insert_cohesive_interface` for 8-node hexahedral meshes: the
    crack path is a set of interior quadrilateral faces (given by their 4 node ids in cyclic
    order). The same connected-components node-duplication logic applies with element
    adjacency through non-crack *faces*. Returned cohesive connectivity rows have 8 entries
    [b0, b1, b2, b3, t0, t1, t2, t3] with t_i the duplicate of b_i (the convention of
    elements/cohesive3d.py), and the bottom-face cyclic order is chosen so that the face
    normal (t_xi x t_eta) points from the bottom-side element toward the top-side element --
    i.e. positive normal separation means opening."""
    elements = elements.clone()
    orig_elements = elements.clone()  # pre-duplication connectivity (for centroid orientation)
    n_elem = elements.shape[0]
    crack_face_list = [tuple(f) for f in crack_faces]
    crack_face_set = {_face_key(f) for f in crack_face_list}
    crack_nodes = sorted({n for f in crack_face_set for n in f})

    elem_faces: List[List[Tuple[int, ...]]] = []
    face_to_elems: Dict[Tuple[int, ...], List[int]] = {}
    for ei in range(n_elem):
        nodes = elements[ei].tolist()
        faces = [_face_key([nodes[i] for i in pattern]) for pattern in _HEX_FACES]
        elem_faces.append(faces)
        for f in faces:
            face_to_elems.setdefault(f, []).append(ei)

    # node -> incident elements (avoid the O(n_elem * n_crack_nodes) scan of the 2D version)
    node_elems: Dict[int, List[int]] = {}
    for ei in range(n_elem):
        for n in set(elements[ei].tolist()):
            node_elems.setdefault(n, []).append(ei)

    new_points_list = [points]
    next_point_idx = points.shape[0]
    node_side_map: Dict[int, Dict[int, int]] = {}

    for node in crack_nodes:
        incident_elems = node_elems.get(node, [])
        adjacency = {ei: set() for ei in incident_elems}
        for ei in incident_elems:
            for f in elem_faces[ei]:
                if f in crack_face_set:
                    continue
                for ej in face_to_elems[f]:
                    if ej != ei and ej in adjacency:
                        adjacency[ei].add(ej)

        visited = set()
        groups: List[List[int]] = []
        for ei in incident_elems:
            if ei in visited:
                continue
            stack, comp = [ei], []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                stack.extend(adjacency[cur] - visited)
            groups.append(sorted(comp))
        groups.sort(key=lambda g: g[0])

        mapping = {}
        for gi, group in enumerate(groups):
            if gi == 0:
                node_id = node
            else:
                node_id = next_point_idx
                next_point_idx += 1
                new_points_list.append(points[node : node + 1])
            for ei in group:
                mapping[ei] = node_id
        node_side_map[node] = mapping

    for node, mapping in node_side_map.items():
        for ei, node_id in mapping.items():
            row = elements[ei].tolist()
            elements[ei] = torch.tensor([node_id if v == node else v for v in row], dtype=elements.dtype)

    cohesive_rows = []
    for face in crack_face_list:
        key = _face_key(face)
        adj = face_to_elems.get(key, [])
        if len(adj) != 2:
            continue  # boundary face: nothing to bond
        e_lo, e_hi = sorted(adj)
        bottom = [node_side_map[n][e_lo] for n in face]
        top = [node_side_map[n][e_hi] for n in face]

        # Orient so the face normal points from e_lo toward e_hi (opening positive).
        p = points
        v1 = p[face[1]] - p[face[0]]
        v2 = p[face[3]] - p[face[0]]
        normal = torch.linalg.cross(v1, v2)
        centroid_face = p[list(face)].mean(dim=0)
        centroid_hi = p[orig_elements[e_hi]].mean(dim=0)
        if torch.dot(normal, centroid_hi - centroid_face) < 0:
            bottom = [bottom[0]] + bottom[1:][::-1]
            top = [top[0]] + top[1:][::-1]
        cohesive_rows.append(bottom + top)

    new_points = torch.cat(new_points_list, dim=0)
    cohesive_connectivity = torch.tensor(cohesive_rows, dtype=torch.long)
    return CohesiveInsertionResult(points=new_points, elements=elements, cohesive_connectivity=cohesive_connectivity)
