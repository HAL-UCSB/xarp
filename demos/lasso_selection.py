import numpy as np
from PIL import Image, ImageDraw

from xarp import colors
from xarp.data_models import CameraIntrinsics, Hands
from xarp.entities import DefaultAssets, Element, ImageAsset
from xarp.express import SyncXR
from xarp.gestures import THUMB_TIP, pinch
from xarp.server import make_qrcode_image, run
from xarp.spatial import Pose, Transform, Vector3, convex_hull_2d

DISPLAY_DISTANCE = 0.49
POINT_SCALE = Vector3.one() * 0.01
MIN_LASSO_POINTS = 3
MIN_CONVEX_HULL_POINTS = 3
MASK_COLOR = (0, 255, 64, 110)


def lasso_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Return a convex hull for a 2D lasso point cloud.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be of shape (N, 2)")

    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.unique(pts, axis=0)
    if len(pts) < MIN_CONVEX_HULL_POINTS:
        return np.empty((0, 2), dtype=np.float64)

    hull = convex_hull_2d(pts)
    return hull if len(hull) >= MIN_CONVEX_HULL_POINTS else np.empty((0, 2), dtype=np.float64)


def _draw_ordered_points(
        draw: ImageDraw.ImageDraw,
        pts_int: np.ndarray,
        color: tuple[int, int, int, int],
        width: int,
        height: int,
) -> None:
    if len(pts_int) == 1:
        u, v = map(int, pts_int[0])
        if 0 <= u < width and 0 <= v < height:
            draw.point((u, v), fill=color)
    elif len(pts_int) == 2:
        draw.line([tuple(uv) for uv in pts_int], fill=color, width=4)
    else:
        draw.polygon([tuple(uv) for uv in pts_int], fill=color)


def draw_mask(
        points: np.ndarray,
        image_size: tuple[int, int],
        color: tuple[int, int, int, int] = MASK_COLOR,
) -> Image.Image:
    """
    Draw a transparent RGBA overlay from lasso points in image-pixel space.
    """
    width, height = image_size
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be of shape (N, 2)")
    pts = pts[np.isfinite(pts).all(axis=1)]

    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if len(pts) == 0:
        return mask

    draw = ImageDraw.Draw(mask)
    if len(pts) < MIN_CONVEX_HULL_POINTS:
        _draw_ordered_points(draw, np.rint(pts).astype(np.int32), color, width, height)
    else:
        hull = lasso_hull_2d(pts)
        fill_points = hull if len(hull) >= 3 else pts
        _draw_ordered_points(draw, np.rint(fill_points).astype(np.int32), color, width, height)

    return mask


def lasso_to_pixels(
        lasso: list[Vector3],
        eye: Pose,
        image_size: tuple[int, int],
        intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Project world-space lasso points into the current camera image.
    """
    width, height = image_size
    pixels = [
        intrinsics.world_point_to_panel_pixel(point, eye, width, height)
        for point in lasso
    ]
    pixels = np.asarray(pixels, dtype=np.float64)
    return pixels[np.isfinite(pixels).all(axis=1)]


def selected_image(
        img: ImageAsset,
        lasso: list[Vector3],
        eye: Pose,
        intrinsics: CameraIntrinsics,
) -> ImageAsset:
    source = img.obj.convert("RGBA")
    lasso_pixels = lasso_to_pixels(lasso, eye, source.size, intrinsics)
    mask = draw_mask(lasso_pixels, source.size)
    return ImageAsset.from_obj(Image.alpha_composite(source, mask))


def image_panel_for_eye(img: ImageAsset, eye: Pose) -> Element:
    return Element(
        key="lasso_image_panel",
        asset=img,
        transform=Transform(
            position=eye.ray_point(DISPLAY_DISTANCE),
            rotation=eye.rotation,
        ),
    )


def app(xr: SyncXR, *args, **kwargs) -> None:
    xr.write("Lasso Selection")
    xr.destroy_element(all_elements=True)
    intrinsics = xr.info().rgb_intrinsics

    cursor = Element(
        key="lasso_cursor",
        asset=DefaultAssets.sphere(),
        transform=Transform(scale=POINT_SCALE),
        color=colors.GRAY,
    )
    lasso_point = Element(
        key="lasso_point",
        asset=DefaultAssets.sphere(),
        transform=Transform(scale=POINT_SCALE),
        color=colors.GREEN,
    )

    lasso: list[Vector3] = []
    point_index = 0
    was_committing = False

    stream = xr.sense(image=True, eye=True, hands=True)
    for frame in stream:
        img: ImageAsset = frame["image"]
        eye: Pose = frame["eye"]
        hands: Hands = frame["hands"]

        if hands.right:
            cursor.transform.position = hands.right[THUMB_TIP].position
            xr.update(cursor)

            if pinch(hands.right):
                point = hands.right[THUMB_TIP].position
                lasso.append(point)
                lasso_point.key = f"lasso_point_{point_index}"
                lasso_point.transform.position = point
                point_index += 1
                xr.update(lasso_point)

        committing = bool(hands.left and pinch(hands.left))
        if committing and not was_committing:
            panel = None
            if len(lasso) >= MIN_LASSO_POINTS:
                panel = image_panel_for_eye(selected_image(img, lasso, eye, intrinsics), eye)

            xr.destroy_element(all_elements=True)
            if panel is not None:
                xr.update(panel)

            lasso.clear()
            point_index = 0

        was_committing = committing

    stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
