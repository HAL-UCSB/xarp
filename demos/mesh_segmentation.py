from enum import Enum

import numpy as np
import requests
import trimesh

from xarp import colors
from xarp.data_models import Hands
from xarp.entities import DefaultAssets, Element, GLBAsset
from xarp.express import SyncXR
from xarp.gestures import THUMB_TIP, pinch
from xarp.server import make_qrcode_image, run
from xarp.spatial import Quaternion, Transform, Vector3, Vector4

glb_url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"
response = requests.get(
    glb_url,
    headers={"User-Agent": "python"},
    timeout=10,
)
response.raise_for_status()
glb_bytes = response.content

PAINT_RADIUS = 0.025
PAINT_ALPHA = 0.35
BRUSH_SCALE = Vector3.one() * 0.02
MESH_SCALE = Vector3.one() * 0.1
KEEP_PAINT_COLOR = Vector4(colors.GREEN.x, colors.GREEN.y, colors.GREEN.z, PAINT_ALPHA)
REMOVE_PAINT_COLOR = Vector4(colors.RED.x, colors.RED.y, colors.RED.z, PAINT_ALPHA)
TRIMESH_TO_DISPLAY_MESH = np.diag([-1.0, 1.0, 1.0, 1.0])


class PaintMode(Enum):
    KEEP = "keep"
    REMOVE = "remove"


def faces_intersecting_world_spheres(
        mesh: trimesh.Trimesh,
        mesh_to_world: np.ndarray,
        world_centers: np.ndarray,
        world_radius: float,
) -> np.ndarray:
    vertices_h = np.column_stack((
        mesh.vertices,
        np.ones(len(mesh.vertices), dtype=np.float64),
    ))
    world_vertices = (mesh_to_world @ vertices_h.T).T[:, :3]
    triangles = world_vertices[mesh.faces]
    radius_sq = world_radius * world_radius
    near_faces = np.zeros(len(mesh.faces), dtype=bool)

    for center in world_centers:
        near_faces |= point_triangle_distance_squared(center, triangles) <= radius_sq

    return near_faces


def point_triangle_distance_squared(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    ab = b - a
    ac = c - a
    ap = point - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    result = np.empty(len(triangles), dtype=np.float64)
    remaining = np.ones(len(triangles), dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    result[mask] = np.einsum("ij,ij->i", ap[mask], ap[mask])
    remaining &= ~mask

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = remaining & (d3 >= 0.0) & (d4 <= d3)
    result[mask] = np.einsum("ij,ij->i", bp[mask], bp[mask])
    remaining &= ~mask

    vc = d1 * d4 - d3 * d2
    mask = remaining & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = d1[mask] / (d1[mask] - d3[mask])
    closest = a[mask] + ab[mask] * v[:, None]
    delta = point - closest
    result[mask] = np.einsum("ij,ij->i", delta, delta)
    remaining &= ~mask

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = remaining & (d6 >= 0.0) & (d5 <= d6)
    result[mask] = np.einsum("ij,ij->i", cp[mask], cp[mask])
    remaining &= ~mask

    vb = d5 * d2 - d1 * d6
    mask = remaining & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = d2[mask] / (d2[mask] - d6[mask])
    closest = a[mask] + ac[mask] * w[:, None]
    delta = point - closest
    result[mask] = np.einsum("ij,ij->i", delta, delta)
    remaining &= ~mask

    va = d3 * d6 - d5 * d4
    mask = remaining & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
    closest = b[mask] + (c[mask] - b[mask]) * w[:, None]
    delta = point - closest
    result[mask] = np.einsum("ij,ij->i", delta, delta)
    remaining &= ~mask

    denom = 1.0 / (va[remaining] + vb[remaining] + vc[remaining])
    v = vb[remaining] * denom
    w = vc[remaining] * denom
    closest = a[remaining] + ab[remaining] * v[:, None] + ac[remaining] * w[:, None]
    delta = point - closest
    result[remaining] = np.einsum("ij,ij->i", delta, delta)
    return result


def submesh_from_face_mask(mesh: trimesh.Trimesh, near_mask: np.ndarray) -> trimesh.Trimesh:
    near_face_indices = np.where(near_mask)[0]
    if len(near_face_indices) == 0:
        return trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
    return mesh.submesh([near_face_indices], append=True)


def points_to_array(points: list[Vector3]) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)


def mesh_to_world_matrix(element_transform: Transform) -> np.ndarray:
    return element_transform.to_matrix() @ TRIMESH_TO_DISPLAY_MESH


def make_paint_element(key: str, position: Vector3, mode: PaintMode) -> Element:
    return Element(
        key=key,
        asset=DefaultAssets.sphere(),
        transform=Transform(position=position, scale=BRUSH_SCALE),
        color=KEEP_PAINT_COLOR if mode is PaintMode.KEEP else REMOVE_PAINT_COLOR,
    )


def random_transform(head_transform: Transform) -> Transform:
    forward_distance = np.random.uniform(0.25, 0.45)
    lateral_offset = np.random.uniform(-0.12, 0.12)
    vertical_offset = np.random.uniform(-0.08, 0.08)
    position = (
        head_transform.ray_point(float(forward_distance))
        + head_transform.right * float(lateral_offset)
        + head_transform.up * float(vertical_offset)
    )
    rotation = Quaternion.from_euler_angles(
        float(np.random.uniform(-35.0, 35.0)),
        float(np.random.uniform(0.0, 360.0)),
        float(np.random.uniform(-35.0, 35.0)),
    )
    return Transform(position=position, rotation=rotation, scale=MESH_SCALE)


def update_mesh(
        xr: SyncXR,
        mesh: Element,
        full_mesh: trimesh.Trimesh,
        working_face_mask: np.ndarray,
        path: list[Vector3],
        mode: PaintMode,
        asset_index: int,
) -> tuple[np.ndarray, int]:
    mesh_to_world = mesh_to_world_matrix(mesh.transform)
    world_path = points_to_array(path)
    near_mask = faces_intersecting_world_spheres(full_mesh, mesh_to_world, world_path, PAINT_RADIUS)

    if mode is PaintMode.KEEP:
        next_face_mask = working_face_mask | near_mask
    else:
        next_face_mask = working_face_mask & ~near_mask

    next_mesh = submesh_from_face_mask(full_mesh, next_face_mask)
    next_mesh.remove_unreferenced_vertices()
    next_asset = GLBAsset.from_obj(next_mesh, asset_key=f"mesh_asset_{asset_index}")
    xr.save(next_asset)
    next_asset.raw = None
    mesh.asset = next_asset
    xr.update(mesh)
    return next_face_mask, asset_index + 1


def app(xr: SyncXR, *args, **kwargs) -> None:
    xr.destroy_asset(all_assets=True)
    xr.destroy_element(all_elements=True)

    source_asset = GLBAsset(raw=glb_bytes)
    full_mesh = source_asset.obj
    working_face_mask = np.ones(len(full_mesh.faces), dtype=bool)
    base_asset = GLBAsset.from_obj(full_mesh, asset_key="mesh_asset_0")
    xr.save(base_asset)
    base_asset.raw = None

    mesh = Element(
        key="mesh",
        asset=base_asset,
        transform=random_transform(xr.head()),
    )
    xr.update(mesh)

    paint_mode: PaintMode | None = None
    paint_points: list[Vector3] = []
    paint_elements: list[Element] = []
    paint_index = 0
    asset_index = 1

    stream = xr.sense(hands=True)
    for frame in stream:
        hands: Hands = frame["hands"]

        if not hands.left and not hands.right:
            if paint_mode is not None and paint_points:
                working_face_mask, asset_index = update_mesh(
                    xr,
                    mesh,
                    full_mesh,
                    working_face_mask,
                    paint_points,
                    paint_mode,
                    asset_index,
                )
                xr.destroy_element(paint_elements)
                paint_mode = None
                paint_points = []
                paint_elements = []
            continue

        active_hand = None
        active_mode = None
        if hands.right and pinch(hands.right):
            active_hand = hands.right
            active_mode = PaintMode.KEEP
        elif hands.left and pinch(hands.left):
            active_hand = hands.left
            active_mode = PaintMode.REMOVE

        if active_hand is None:
            continue

        if paint_mode is not active_mode:
            xr.destroy_element(paint_elements)
            paint_mode = active_mode
            paint_points = []
            paint_elements = []
            paint_index = 0

        paint_position = active_hand[THUMB_TIP].position
        paint_points.append(paint_position)
        paint = make_paint_element(f"paint_{paint_mode.value}_{paint_index}", paint_position, paint_mode)
        paint_elements.append(paint)
        paint_index += 1
        xr.update(paint)

    stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
