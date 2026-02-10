import math
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from pydantic import RootModel


class Vector3(RootModel[list[float]]):
    model_config = ConfigDict(
        validate_assignment=True
    )

    @property
    def x(self) -> float:
        return self.root[0]

    @x.setter
    def x(self, value: float) -> None:
        self.root[0] = float(value)

    @property
    def y(self) -> float:
        return self.root[1]

    @y.setter
    def y(self, value: float) -> None:
        self.root[1] = float(value)

    @property
    def z(self) -> float:
        return self.root[2]

    @z.setter
    def z(self, value: float) -> None:
        self.root[2] = float(value)

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> Self:
        return cls([float(x), float(y), float(z)])

    @staticmethod
    def zero() -> Self:
        return Vector3([0.0, 0.0, 0.0])

    @staticmethod
    def one() -> Self:
        return Vector3([1.0, 1.0, 1.0])

    @staticmethod
    def forward() -> Self:
        return Vector3([0, 0, 1.0])

    @staticmethod
    def up() -> Self:
        return Vector3([0, 1.0, 0])

    @staticmethod
    def right() -> Self:
        return Vector3([1.0, 0, 0])

    @staticmethod
    def left() -> Self:
        return Vector3([-1.0, 0, 0])

    def to_numpy(self) -> np.ndarray:
        return np.array(self.root, dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.root))

    def normalized(self) -> Self:
        arr = self.to_numpy()
        n = np.linalg.norm(arr)
        if n == 0.0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3((arr / n).tolist())

    def project_on_plane(self, on_plane: Self, normal: Self) -> Self:
        n = normal.to_numpy()
        nn = float(np.dot(n, n))
        if nn == 0.0:
            raise ValueError("Plane normal cannot be zero vector")
        v = self.to_numpy()
        p = on_plane.to_numpy()
        projected = v - n * (np.dot(v - p, n) / nn)
        return Vector3(projected.tolist())

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3([
            self.root[0] + other.root[0],
            self.root[1] + other.root[1],
            self.root[2] + other.root[2],
        ])

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3([
            self.root[0] - other.root[0],
            self.root[1] - other.root[1],
            self.root[2] - other.root[2],
        ])

    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __rsub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __mul__(self, scalar: float) -> Self:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector3([
            self.root[0] * scalar,
            self.root[1] * scalar,
            self.root[2] * scalar,
        ])

    def __rmul__(self, scalar: float) -> Self:
        return self.__mul__(scalar)


class Quaternion(RootModel[list[float]]):
    model_config = ConfigDict(
        validate_assignment=True
    )

    @classmethod
    def from_xyzw(cls, x: float, y: float, z: float, w: float) -> Self:
        return cls([float(x), float(y), float(z), float(w)])

    @property
    def x(self) -> float:
        return self.root[0]

    @property
    def y(self) -> float:
        return self.root[1]

    @property
    def z(self) -> float:
        return self.root[2]

    @property
    def w(self) -> float:
        return self.root[3]

    def to_numpy(self) -> np.ndarray:
        # Returns in xyzw order
        return np.array(self.root, dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.root))

    def normalized(self) -> Self:
        arr = self.to_numpy()
        n = np.linalg.norm(arr)
        if n == 0.0:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion((arr / n).tolist())

    @staticmethod
    def zero() -> Self:
        return Quaternion([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def identity() -> Self:
        return Quaternion([0.0, 0.0, 0.0, 1.0])

    def to_euler_angles(self, degrees=True) -> Vector3:
        """
        Returns XYZ Euler angles (roll, pitch, yaw) in radians.
        """
        q = self.normalized()
        x, y, z, w = q.x, q.y, q.z, q.w

        # Roll (X-axis)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)  # gimbal lock
        else:
            pitch = math.asin(sinp)

        # Yaw (Z-axis)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if degrees:
            return Vector3([
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw),
            ])
        return Vector3([roll, pitch, yaw])

    @classmethod
    def from_euler_angles(
            cls,
            roll: float,
            pitch: float,
            yaw: float,
            degrees: bool = True,
    ) -> Self:
        """
        Create a quaternion from XYZ Euler angles (roll, pitch, yaw).

        The rotation order is intrinsic X → Y → Z (roll, pitch, yaw),
        consistent with `to_euler_angles`.

        Angles are interpreted as degrees by default.
        """
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        hr = roll * 0.5
        hp = pitch * 0.5
        hy = yaw * 0.5

        cr = math.cos(hr)
        sr = math.sin(hr)
        cp = math.cos(hp)
        sp = math.sin(hp)
        cy = math.cos(hy)
        sy = math.sin(hy)

        # XYZ intrinsic rotation (roll → pitch → yaw)
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy

        return cls.from_xyzw(x, y, z, w).normalized()

    def to_matrix(self) -> np.ndarray:
        """
        Returns a 3x3 rotation matrix.
        right-handed rotation.
        """
        q = self.normalized()
        x, y, z, w = q.x, q.y, q.z, q.w

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=float)

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Self:
        """
        Create a quaternion from a 3x3 right-handed rotation matrix.

        Expects m to be a proper rotation matrix (orthonormal, det ~ +1).
        """
        m = np.asarray(m, dtype=float)
        if m.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        trace = float(m[0, 0] + m[1, 1] + m[2, 2])

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        return cls.from_xyzw(x, y, z, w).normalized()

    def inverse(self) -> Self:
        """
        Returns the inverse of the quaternion.
        For unit quaternions, this is just the conjugate.
        """
        x, y, z, w = self.x, self.y, self.z, self.w
        n2 = x * x + y * y + z * z + w * w
        if n2 == 0.0:
            raise ValueError("Cannot invert zero quaternion")
        return Quaternion([
            -x / n2,
            -y / n2,
            -z / n2,
            w / n2,
        ])

    @classmethod
    def from_up_forward(cls, up: Vector3, forward: Vector3 = None) -> Self:
        """
        Create a quaternion from desired up (+Y) and forward (+Z) directions.

        Right-handed frame:
            right = up × forward
        Columns of rotation matrix are [right, up, forward].
        Default forward is +Z
        """
        if forward is None:
            forward = Vector3.forward()

        u = up.normalized().to_numpy()
        f = forward.normalized().to_numpy()

        r = np.cross(u, f)
        rn = float(np.linalg.norm(r))
        if rn == 0.0:
            raise ValueError("up and forward vectors must not be collinear")
        r /= rn

        # re-orthogonalize forward to eliminate drift
        f = np.cross(r, u)

        m = np.column_stack((r, u, f))  # columns: right, up, forward
        return cls.from_matrix(m)

    def rotate_by_euler(
            self,
            roll: float = 0,
            pitch: float = 0,
            yaw: float = 0,
            degrees: bool = True,
    ) -> Self:
        """
        Rotate this quaternion by the given Euler angles (roll, pitch, yaw).

        Returns a new quaternion representing the combined rotation.
        The rotation is applied in the quaternion's local frame.

        Args:
            roll: Rotation around X-axis
            pitch: Rotation around Y-axis
            yaw: Rotation around Z-axis
            degrees: Whether angles are in degrees (True) or radians (False)

        Returns:
            New quaternion with the rotation applied
        """
        rotation = Quaternion.from_euler_angles(roll, pitch, yaw, degrees=degrees)
        return self * rotation

    def __mul__(self, other: Self) -> Self:
        """
        Quaternion multiplication (Hamilton product).
        Returns self * other, representing applying other's rotation after self's.
        """
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply with another Quaternion")

        x1, y1, z1, w1 = self.x, self.y, self.z, self.w
        x2, y2, z2, w2 = other.x, other.y, other.z, other.w

        return Quaternion.from_xyzw(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )


class Pose(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

    position: Vector3 = Field(default_factory=Vector3.zero)
    rotation: Quaternion = Field(default_factory=Quaternion.identity)

    def to_matrix(self) -> np.ndarray:
        # Scale matrix
        S = np.eye(4)
        S[0, 0] = 1
        S[1, 1] = 1
        S[2, 2] = 1

        # Rotation matrix
        R = self.rotation.to_matrix()

        # Translation matrix
        T = np.eye(4)
        T[:3, 3] = self.position.to_numpy()

        # Combine (T * R * S)
        return T @ R @ S

    def ray(self, d: float) -> Vector3:
        forward = Vector3.forward()
        R = self.rotation.to_matrix()  # 3x3
        world_dir = R @ forward.to_numpy()
        return Vector3(self.position.to_numpy() + d * world_dir)


class Transform(Pose):
    parent: str | None = None
    scale: Vector3 = Field(default_factory=Vector3.one)

    def to_matrix(self) -> np.ndarray:
        matrix = super().to_matrix()
        matrix[0, 0] = self.scale[0]
        matrix[1, 1] = self.scale[1]
        matrix[2, 2] = self.scale[2]
        return matrix

    def world_matrix(self) -> np.ndarray:
        M = self.to_matrix()
        if self.parent is not None:
            return self.parent.world_matrix() @ M
        return M


def distance(a: Vector3, b: Vector3):
    return np.linalg.norm(a.to_numpy() - b.to_numpy())


def angle_quaternion(q1: Quaternion, q2: Quaternion, degrees: bool = True) -> float:
    q1 = q1.normalized()
    q2 = q2.normalized()

    # Relative rotation: q_rel = q1^{-1} * q2
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w

    # Quaternion multiplication (inverse(q1) * q2)
    w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    x = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    y = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    z = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2

    # Clamp for numerical safety
    w = max(-1.0, min(1.0, abs(w)))

    angle = 2.0 * math.acos(w)

    if degrees:
        return math.degrees(angle)
    return angle


def cosine_similarity(a: Vector3, b: Vector3):
    a = a.to_numpy()
    b = b.to_numpy()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 1:
        return pts.copy()

    # Sort by x, then by y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        # 2D cross product (OA x OB)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    # Concatenate lower + upper, removing the last point of each (they repeat)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def point_to_ray_distance(point: Vector3, pose: Pose) -> float:
    local_ray_dir = Vector3.from_xyz(0, 0, 1)

    o = pose.position.to_numpy()
    p = point.to_numpy()

    # World direction from pose rotation
    R = pose.rotation.to_matrix()
    d = R @ local_ray_dir.to_numpy()

    d_norm = float(np.linalg.norm(d))
    if d_norm == 0.0:
        raise ValueError("Ray direction is zero after rotation (invalid local_ray_dir or rotation).")
    d = d / d_norm

    v = p - o
    t = float(np.dot(v, d))  # projection onto direction

    if t <= 0.0:
        # closest point is the ray origin
        return float(np.linalg.norm(v))

    # closest point is o + t d
    closest = o + t * d
    return float(np.linalg.norm(p - closest))
