import math

from PIL import Image

from xarp.data_models import Hands
from xarp.entities import Element, ImageAsset
from xarp.express import SyncXR, AsyncGeneratorIterator
from xarp.gestures import INDEX_TIP, THUMB_METACARPAL, THUMB_TIP, index_thumb_l
from xarp.server import run, make_qrcode_image
from xarp.spatial import Quaternion, Vector3, Pose

CAMERA_LOWER_CORNER_JOINT = THUMB_METACARPAL
CAMERA_UPPER_CORNER_JOINT = THUMB_TIP
SUPPORT_HAND_PADDING = 0.04


def ray_plane_intersection(eye_pose: Pose, ray: Vector3, focal_distance: float = .49) -> Vector3:
    """
    Intersect a world-space ray with the camera-local image plane.

    The ray starts at eye_pose.position and points in the world-space direction
    given by ray. The returned point is in eye/camera-local coordinates on the
    plane z == focal_distance.

    :param eye_pose: Camera/eye pose defining the ray origin and camera axes.
    :param ray: World-space ray direction.
    :param focal_distance: Positive distance from the eye to the image plane.
    :return: Camera-local intersection point on the image plane.
    """
    if focal_distance <= 0.0:
        raise ValueError("focal_distance must be positive")

    local_ray = eye_pose.inverse_transform_vector(ray)
    if math.isclose(local_ray.z, 0.0, abs_tol=1e-12):
        raise ValueError("ray is parallel to the image plane")

    t = focal_distance / local_ray.z
    if t <= 0.0:
        raise ValueError("ray points away from the image plane")

    return local_ray * t


def crop_image_rectangle(
        img: ImageAsset | Image.Image,
        image_lower_left: Vector3,
        image_upper_right: Vector3,
) -> ImageAsset | Image.Image:
    """
    Crop an axis-aligned image rectangle from two camera-plane corners.

    The corners are camera-local points returned by ray_plane_intersection.
    They are projected to normalized image coordinates using x/z and y/z, then
    mapped to pixels. The crop bounds are clipped to the source image.

    :param img: ImageAsset or PIL image to crop.
    :param image_lower_left: One camera-plane corner of the crop.
    :param image_upper_right: Opposite camera-plane corner of the crop.
    :return: Cropped image, preserving the input image wrapper type.
    """
    source = img.obj if isinstance(img, ImageAsset) else img
    if not isinstance(source, Image.Image):
        raise TypeError("img must be an ImageAsset or PIL Image")

    width, height = source.size

    def to_pixel(point: Vector3) -> tuple[float, float]:
        if math.isclose(point.z, 0.0, abs_tol=1e-12):
            raise ValueError("camera-plane point must have a non-zero z value")
        u = (0.5 + point.x / (2.0 * point.z)) * width
        v = (0.5 - point.y / (2.0 * point.z)) * height
        return u, v

    u1, v1 = to_pixel(image_lower_left)
    u2, v2 = to_pixel(image_upper_right)

    left = max(0, math.floor(min(u1, u2)))
    upper = max(0, math.floor(min(v1, v2)))
    right = min(width, math.ceil(max(u1, u2)))
    lower = min(height, math.ceil(max(v1, v2)))

    if left >= right or upper >= lower:
        raise ValueError("crop rectangle does not overlap the image")

    cropped = source.crop((left, upper, right, lower))
    if isinstance(img, ImageAsset):
        return ImageAsset.from_obj(cropped, mime_type=img.mime_type)
    return cropped


def _source_image(img: ImageAsset | Image.Image) -> Image.Image:
    source = img.obj if isinstance(img, ImageAsset) else img
    if not isinstance(source, Image.Image):
        raise TypeError("img must be an ImageAsset or PIL Image")
    return source


def _wrap_image(source: Image.Image, img: ImageAsset | Image.Image) -> ImageAsset | Image.Image:
    if isinstance(img, ImageAsset):
        return ImageAsset.from_obj(source, mime_type=img.mime_type)
    return source


def _camera_plane_to_pixel(point: Vector3, width: int, height: int) -> tuple[float, float]:
    if math.isclose(point.z, 0.0, abs_tol=1e-12):
        raise ValueError("camera-plane point must have a non-zero z value")
    u = (0.5 + point.x / (2.0 * point.z)) * width
    v = (0.5 - point.y / (2.0 * point.z)) * height
    return u, v


def _projected_aperture_up(eye_pose: Pose, support_hand) -> Vector3:
    index_axis = support_hand[INDEX_TIP].position - support_hand[THUMB_METACARPAL].position
    forward = eye_pose.forward

    # Remove the forward component so the index direction becomes an in-plane
    # up vector for the panel.
    up = index_axis - forward * index_axis.dot(forward)
    if up.norm() < 1e-6:
        up = eye_pose.up - forward * eye_pose.up.dot(forward)
    if up.norm() < 1e-6:
        up = Vector3.up() - forward * Vector3.up().dot(forward)
    if up.norm() < 1e-6:
        up = Vector3.right() - forward * Vector3.right().dot(forward)

    return up.normalized()


def crop_image_aperture(
        img: ImageAsset | Image.Image,
        image_corner_a: Vector3,
        image_corner_b: Vector3,
        eye_pose: Pose,
        support_hand,
) -> ImageAsset | Image.Image:
    """
    Sample the hand aperture as a sliding window over the original camera image.

    image_corner_a and image_corner_b are opposite corners on the camera-local
    image plane. The support hand's L shape defines the window size and texture
    basis so the rotated panel does not rotate the camera image content in
    world space.
    """
    source = _source_image(img)
    image_width, image_height = source.size

    if image_corner_a.z <= 0.0 or image_corner_b.z <= 0.0:
        raise ValueError("aperture rectangle must be in front of the camera")

    aperture_up_world = _projected_aperture_up(eye_pose, support_hand)
    aperture_up = eye_pose.inverse_transform_vector(aperture_up_world).normalized()
    center_plane = (image_corner_a + image_corner_b) * 0.5
    center_u, center_v = _camera_plane_to_pixel(center_plane, image_width, image_height)
    corner_a_u, corner_a_v = _camera_plane_to_pixel(image_corner_a, image_width, image_height)
    corner_b_u, corner_b_v = _camera_plane_to_pixel(image_corner_b, image_width, image_height)

    # Build the sampling frame in image-pixel space. This avoids shear from
    # non-square camera buffers where camera-plane X and Y scale differently.
    pixel_scale_x = image_width / (2.0 * center_plane.z)
    pixel_scale_y = image_height / (2.0 * center_plane.z)
    up_u = aperture_up.x * pixel_scale_x
    up_v = -aperture_up.y * pixel_scale_y
    up_norm = math.hypot(up_u, up_v)
    if up_norm < 1e-12:
        raise ValueError("aperture up axis is degenerate in image space")

    up_u /= up_norm
    up_v /= up_norm
    right_u = up_v
    right_v = -up_u

    diagonal_u = corner_b_u - corner_a_u
    diagonal_v = corner_b_v - corner_a_v
    output_width = max(1, math.ceil(abs(diagonal_u * right_u + diagonal_v * right_v)))
    output_height = max(1, math.ceil(abs(diagonal_u * up_u + diagonal_v * up_v)))
    if output_width <= 1 or output_height <= 1:
        raise ValueError("aperture rectangle must have non-zero width and height")

    roll = aperture_roll(eye_pose, support_hand)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)

    # One inverse affine does the previous padded-crop + inverse-roll + center
    # crop in a single resample. Output pixels are in panel-local coordinates;
    # source pixels are in the camera image.
    a = cos_roll
    b = sin_roll
    c = center_u - a * (output_width - 1) * 0.5 - b * (output_height - 1) * 0.5
    d = -sin_roll
    e = cos_roll
    f = center_v - d * (output_width - 1) * 0.5 - e * (output_height - 1) * 0.5

    cropped = source.transform(
        (output_width, output_height),
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BILINEAR,
    )
    return _wrap_image(cropped, img)


def aperture_roll(eye_pose: Pose, support_hand) -> float:
    """
    Signed roll from the camera image frame to the hand aperture frame.
    """
    forward = eye_pose.forward
    camera_up = eye_pose.up - forward * eye_pose.up.dot(forward)
    if camera_up.norm() < 1e-6:
        camera_up = Vector3.up() - forward * Vector3.up().dot(forward)
    if camera_up.norm() < 1e-6:
        camera_up = Vector3.right() - forward * Vector3.right().dot(forward)

    camera_up = camera_up.normalized()
    aperture_up = _projected_aperture_up(eye_pose, support_hand)
    sin_theta = forward.dot(camera_up.cross(aperture_up))
    cos_theta = camera_up.dot(aperture_up)
    return math.atan2(sin_theta, cos_theta)


def aperture_frame_rotation(eye_pose: Pose, support_hand) -> Quaternion:
    """
    Build panel rotation from the hand-defined aperture frame.

    The panel keeps the camera's forward direction, but its roll comes from the
    support hand's index-thumb L shape instead of the head-mounted camera roll.
    """
    return Quaternion.from_up_forward(_projected_aperture_up(eye_pose, support_hand), eye_pose.forward)


def app(xr: SyncXR, *args, **kwargs) -> None:
    panel = Element(key="image_panel")

    stream = xr.sense(image=True, eye=True, hands=True)
    for frame in stream:
        img: ImageAsset = frame["image"]
        eye: Pose = frame["eye"]
        hands: Hands = frame["hands"]

        if not hands.left or not hands.right:
            continue

        if hands.right and index_thumb_l(hands.right):
            support_hand = hands.right
            shutter_hand = hands.left
        elif hands.left and index_thumb_l(hands.left):
            support_hand = hands.left
            shutter_hand = hands.right
        else:
            # panel.active = False
            # xr.update(panel)
            continue

        panel.active = True
        raw_support_vertex = support_hand[THUMB_METACARPAL].position
        shutter_vertex = shutter_hand[THUMB_TIP].position
        camera_diagonal = shutter_vertex - raw_support_vertex
        if camera_diagonal.norm() > SUPPORT_HAND_PADDING:
            support_vertex = raw_support_vertex + camera_diagonal.normalized() * SUPPORT_HAND_PADDING
        else:
            support_vertex = raw_support_vertex

        try:
            padded_diagonal = shutter_vertex - support_vertex
            eye_to_support = support_vertex - eye.position
            eye_to_shutter = shutter_vertex - eye.position
            img_lower_corner = ray_plane_intersection(eye, eye_to_support, .49)
            image_upper_corner = ray_plane_intersection(eye, eye_to_shutter, .49)
            img = crop_image_aperture(img, img_lower_corner, image_upper_corner, eye, support_hand)
            panel.asset = img
            panel.transform.position = support_vertex + padded_diagonal * .5
            panel.transform.rotation = aperture_frame_rotation(eye, support_hand)
            xr.update(panel)
        except ValueError:
            continue

    stream.close()


if __name__ == '__main__':
    make_qrcode_image()
    run(app)
