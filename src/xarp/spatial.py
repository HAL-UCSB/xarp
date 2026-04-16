import math
from typing import Any, Iterator, Self

import numpy as np
from pydantic import BaseModel
from pydantic import Field, ConfigDict
from pydantic import RootModel


class Vector3(RootModel[list[float]]):
    """
    Immutable 3D vector with float64 components.

    Serializes as a plain array: [x, y, z]

    Coordinate convention: right-handed, Y-up, Z-forward.
    """

    model_config = {"frozen": True}

    __slots__ = ("_arr",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and not kwargs:
            if len(args) == 3:
                super().__init__(root=[float(args[0]), float(args[1]), float(args[2])])
            elif len(args) == 1:
                seq = args[0]
                if hasattr(seq, "__len__") and len(seq) != 3:
                    raise ValueError(f"Expected 3 components, got {len(seq)}")
                super().__init__(root=[float(seq[0]), float(seq[1]), float(seq[2])])
            else:
                raise ValueError(f"Expected 1 or 3 positional arguments, got {len(args)}")
        elif {"x", "y", "z"} <= kwargs.keys():
            super().__init__(root=[float(kwargs["x"]), float(kwargs["y"]), float(kwargs["z"])])
        else:
            super().__init__(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_arr", np.array(self.root, dtype=np.float64))

    # ------------------------------------------------------------------
    # Named component accessors
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        return self.root[0]

    @property
    def y(self) -> float:
        return self.root[1]

    @property
    def z(self) -> float:
        return self.root[2]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_sequence(cls, seq: list[float] | tuple[float, ...] | np.ndarray) -> Self:
        if len(seq) != 3:
            raise ValueError(f"Expected 3 components, got {len(seq)}")
        return cls(float(seq[0]), float(seq[1]), float(seq[2]))

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> Self:
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> Self:
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def forward(cls) -> Self:
        """+Z axis."""
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def backward(cls) -> Self:
        """-Z axis."""
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def up(cls) -> Self:
        """+Y axis."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def down(cls) -> Self:
        """-Y axis."""
        return cls(0.0, -1.0, 0.0)

    @classmethod
    def right(cls) -> Self:
        """+X axis."""
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def left(cls) -> Self:
        """-X axis."""
        return cls(-1.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        return self._arr.copy()

    def to_list(self) -> list[float]:
        return list(self.root)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.root[0], self.root[1], self.root[2])

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[float]:  # type: ignore[override]
        yield self.root[0]
        yield self.root[1]
        yield self.root[2]

    def __getitem__(self, index: int) -> float:
        return float(self._arr[index])

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> Self:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3.from_sequence(self._arr + other._arr)

    def __sub__(self, other: object) -> Self:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3.from_sequence(self._arr - other._arr)

    def __radd__(self, other: object) -> Self:
        return self.__add__(other)

    def __rsub__(self, other: object) -> Self:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3.from_sequence(other._arr - self._arr)

    def __mul__(self, scalar: float) -> Self:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector3.from_sequence(self._arr * float(scalar))

    def __rmul__(self, scalar: float) -> Self:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Self:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0.0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3.from_sequence(self._arr / float(scalar))

    def __neg__(self) -> Self:
        return Vector3.from_sequence(-self._arr)

    def __abs__(self) -> float:
        return self.norm()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def isclose(self, other: Self, atol: float = 1e-9) -> bool:
        """Approximate equality with an absolute tolerance."""
        return bool(np.allclose(self._arr, other._arr, atol=atol, rtol=0.0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return NotImplemented
        return bool(np.array_equal(self._arr, other._arr))

    def __hash__(self) -> int:
        return hash(self._arr.tobytes())

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Vector3({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    # ------------------------------------------------------------------
    # Vector math
    # ------------------------------------------------------------------

    def norm(self) -> float:
        return float(np.linalg.norm(self._arr))

    def norm_squared(self) -> float:
        return float(np.dot(self._arr, self._arr))

    def normalized(self, epsilon: float = 1e-12) -> Self:
        n = self.norm()
        if n < epsilon:
            raise ValueError("Cannot normalize a zero (or near-zero) vector")
        return Vector3.from_sequence(self._arr / n)

    def dot(self, other: Self) -> float:
        return float(np.dot(self._arr, other._arr))

    def cross(self, other: Self) -> Self:
        return Vector3.from_sequence(np.cross(self._arr, other._arr))

    def project_onto(self, other: Self, epsilon: float = 1e-12) -> Self:
        """Orthogonal projection of self onto other."""
        denom = other.norm_squared()
        if denom < epsilon:
            raise ValueError("Cannot project onto a zero vector")
        return (self.dot(other) / denom) * other

    def project_onto_plane(self, plane_origin: Self, plane_normal: Self) -> Self:
        """
        Project self onto the plane defined by a point and a normal.

        Args:
            plane_origin: Any point lying on the plane.
            plane_normal: The plane's normal vector (need not be unit length).
        """
        n = plane_normal.normalized()
        offset = self - plane_origin
        return plane_origin + (offset - offset.dot(n) * n)

    def distance(self, other: Self) -> float:
        return (self - other).norm()

    def lerp(self, other: Self, t: float) -> Self:
        """Linear interpolation. t=0 → self, t=1 → other."""
        return Vector3.from_sequence(self._arr + (other._arr - self._arr) * t)

    def angle(self, other: Self, epsilon: float = 1e-12) -> float:
        """Angle between two vectors in radians, in [0, π]."""
        denom = self.norm() * other.norm()
        if denom < epsilon:
            raise ValueError("Cannot compute angle involving a zero vector")
        cos_theta = float(np.clip(self.dot(other) / denom, -1.0, 1.0))
        return math.acos(cos_theta)


class Quaternion(RootModel[list[float]]):
    """
    Immutable unit quaternion stored as (x, y, z, w).

    Serializes as a plain array: [x, y, z, w]
    """

    model_config = {"frozen": True}

    __slots__ = ("_arr",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and not kwargs:
            if len(args) == 4:
                super().__init__(root=[float(args[0]), float(args[1]), float(args[2]), float(args[3])])
            elif len(args) == 1:
                seq = args[0]
                if hasattr(seq, "__len__") and len(seq) != 4:
                    raise ValueError(f"Expected 4 components, got {len(seq)}")
                super().__init__(root=[float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])])
            else:
                raise ValueError(f"Expected 1 or 4 positional arguments, got {len(args)}")
        elif {"x", "y", "z", "w"} <= kwargs.keys():
            super().__init__(root=[float(kwargs["x"]), float(kwargs["y"]), float(kwargs["z"]), float(kwargs["w"])])
        else:
            super().__init__(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_arr", np.array(self.root, dtype=np.float64))

    # ------------------------------------------------------------------
    # Named component accessors
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_xyzw(cls, x: float, y: float, z: float, w: float) -> Self:
        return cls(float(x), float(y), float(z), float(w))

    @classmethod
    def from_sequence(cls, seq: list[float] | tuple[float, ...] | np.ndarray) -> Self:
        if len(seq) != 4:
            raise ValueError(f"Expected 4 components, got {len(seq)}")
        return cls(float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> Self:
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def identity(cls) -> Self:
        return cls(0.0, 0.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Returns components in xyzw order."""
        return self._arr.copy()

    def to_list(self) -> list[float]:
        return list(self.root)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.root[0], self.root[1], self.root[2], self.root[3])

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return 4

    def __iter__(self):  # type: ignore[override]
        yield self.root[0]
        yield self.root[1]
        yield self.root[2]
        yield self.root[3]

    def __getitem__(self, index: int) -> float:
        return float(self._arr[index])

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def isclose(self, other: Self, atol: float = 1e-9) -> bool:
        """Approximate equality, accounting for q == -q."""
        return bool(
            np.allclose(self._arr, other._arr, atol=atol, rtol=0.0)
            or np.allclose(self._arr, -other._arr, atol=atol, rtol=0.0)
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Quaternion(x={self.x:.6g}, y={self.y:.6g}, z={self.z:.6g}, w={self.w:.6g})"

    # ------------------------------------------------------------------
    # Quaternion math
    # ------------------------------------------------------------------

    def norm(self) -> float:
        return float(np.linalg.norm(self._arr))

    def norm_squared(self) -> float:
        return float(np.dot(self._arr, self._arr))

    def normalized(self, epsilon: float = 1e-12) -> Self:
        n = self.norm()
        if n < epsilon:
            raise ValueError("Cannot normalize a zero (or near-zero) quaternion")
        return Quaternion.from_sequence(self._arr / n)

    def conjugate(self) -> Self:
        return Quaternion.from_xyzw(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> Self:
        n2 = self.norm_squared()
        if n2 < 1e-12:
            raise ValueError("Cannot invert a zero quaternion")
        return Quaternion.from_xyzw(
            -self.x / n2,
            -self.y / n2,
            -self.z / n2,
            self.w / n2,
        )

    def dot(self, other: Self) -> float:
        return float(np.dot(self._arr, other._arr))

    def __mul__(self, other: object) -> Self:
        """Hamilton product: self * other applies other's rotation after self's."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        x1, y1, z1, w1 = self.x, self.y, self.z, self.w
        x2, y2, z2, w2 = other.x, other.y, other.z, other.w
        return Quaternion.from_xyzw(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def __neg__(self) -> Self:
        return Quaternion.from_sequence(-self._arr)

    # ------------------------------------------------------------------
    # Euler angles
    # ------------------------------------------------------------------

    def to_euler_angles(self, degrees: bool = True) -> Vector3:
        """Returns XYZ Euler angles (roll, pitch, yaw)."""
        q = self.normalized()
        x, y, z, w = q.x, q.y, q.z, q.w

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if degrees:
            return Vector3(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
        return Vector3(roll, pitch, yaw)

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
        Rotation order is intrinsic X → Y → Z, consistent with to_euler_angles.
        """
        if degrees:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

        hr, hp, hy = roll * 0.5, pitch * 0.5, yaw * 0.5
        cr, sr = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(hp), math.sin(hp)
        cy, sy = math.cos(hy), math.sin(hy)

        return cls.from_xyzw(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ).normalized()

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Returns a 3x3 rotation matrix."""
        q = self.normalized()
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Self:
        """Create a quaternion from a 3x3 rotation matrix (orthonormal, det ~ +1)."""
        m = np.asarray(m, dtype=float)
        if m.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            return cls.from_xyzw(
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
                0.25 * s,
            ).normalized()
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            return cls.from_xyzw(
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s,
            ).normalized()
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            return cls.from_xyzw(
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s,
            ).normalized()
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            return cls.from_xyzw(
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
                (m[1, 0] - m[0, 1]) / s,
            ).normalized()

    # ------------------------------------------------------------------
    # Higher-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_up_forward(
            cls,
            up: Vector3,
            forward: Vector3 | None = None,
            right_handed: bool = True,
    ) -> Self:
        """Create a quaternion from desired up (+Y) and forward (+Z) directions."""
        if forward is None:
            forward = Vector3.forward()

        u = up.normalized().to_numpy()
        f = forward.normalized().to_numpy()
        r = np.cross(u, f) if right_handed else np.cross(f, u)

        rn = float(np.linalg.norm(r))
        if rn < 1e-12:
            raise ValueError("up and forward vectors must not be collinear")
        r /= rn
        f = np.cross(r, u) if right_handed else np.cross(u, r)

        return cls.from_matrix(np.column_stack((r, u, f)))

    def rotate_by_euler(
            self,
            roll: float = 0.0,
            pitch: float = 0.0,
            yaw: float = 0.0,
            degrees: bool = True,
    ) -> Self:
        """Rotate this quaternion by Euler angles in the local frame."""
        return self * Quaternion.from_euler_angles(roll, pitch, yaw, degrees=degrees)

    def rotate_vector(self, v: Vector3) -> Vector3:
        return Vector3.from_sequence(self.to_matrix() @ v.to_numpy())


class Pose(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    position: Vector3 = Field(default_factory=Vector3.zero)
    rotation: Quaternion = Field(default_factory=Quaternion.identity)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Self:
        """Pose at the origin with no rotation."""
        return cls()

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Self:
        """
        Extract a Pose from a 4×4 rigid-body matrix (no scale, no shear).
        The upper-left 3×3 must be a valid rotation matrix.
        """
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Expected a 4×4 matrix, got shape {m.shape}")
        return cls(
            position=Vector3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
            rotation=Quaternion.from_matrix(m[:3, :3]),
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """4×4 rigid-body matrix [R | t] in float64."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation.to_matrix()
        m[:3, 3] = self.position.to_numpy()
        return m

    def inverse_matrix(self) -> np.ndarray:
        """
        Inverse of the rigid-body matrix.

        For a pure rotation+translation matrix the inverse is cheap:
            R^-1 = R^T,  t^-1 = -R^T @ t
        avoiding a full np.linalg.inv call.
        """
        r = self.rotation.to_matrix()  # 3×3
        t = self.position.to_numpy()  # (3,)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r.T
        m[:3, 3] = -(r.T @ t)
        return m

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __matmul__(self, other: object) -> Self:
        """
        Compose two poses: (self @ other) applies other first, then self.

            position = self.rotation.rotate(other.position) + self.position
            rotation = self.rotation * other.rotation
        """
        if not isinstance(other, Pose):
            return NotImplemented
        return Pose(
            position=self.rotation.rotate_vector(other.position) + self.position,
            rotation=self.rotation * other.rotation,
        )

    def inverse(self) -> Self:
        """The pose that undoes this one: self @ self.inverse() == identity."""
        inv_rotation = self.rotation.inverse()
        inv_position = inv_rotation.rotate_vector(-self.position)
        return Pose(position=inv_position, rotation=inv_rotation)

    # ------------------------------------------------------------------
    # Applying the pose to geometry
    # ------------------------------------------------------------------

    def transform_point(self, v: Vector3) -> Vector3:
        """Rotate then translate a point: R*v + t."""
        r = self.to_matrix() @ np.array([v.x, v.y, v.z, 1.0], dtype=np.float64)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    def transform_vector(self, v: Vector3) -> Vector3:
        """Rotate a direction vector (no translation)."""
        r = self.to_matrix()[:3, :3] @ v.to_numpy()
        return Vector3.from_sequence(r)

    def inverse_transform_point(self, v: Vector3) -> Vector3:
        """Map a world-space point into this pose's local frame."""
        r = self.inverse_matrix() @ np.array([v.x, v.y, v.z, 1.0], dtype=np.float64)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    def inverse_transform_vector(self, v: Vector3) -> Vector3:
        """Map a world-space direction into this pose's local frame."""
        r = self.inverse_matrix()[:3, :3] @ v.to_numpy()
        return Vector3.from_sequence(r)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def forward(self) -> Vector3:
        """Unit vector pointing in the +Z direction of this pose."""
        return self.rotation.rotate_vector(Vector3.forward())

    @property
    def up(self) -> Vector3:
        """Unit vector pointing in the +Y direction of this pose."""
        return self.rotation.rotate_vector(Vector3.up())

    @property
    def right(self) -> Vector3:
        """Unit vector pointing in the +X direction of this pose."""
        return self.rotation.rotate_vector(Vector3.right())

    def ray_point(self, d: float) -> Vector3:
        """World-space point at distance d along the forward direction."""
        return self.position + self.forward * d

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Pose(position={self.position!r}, rotation={self.rotation!r})"


class Transform(Pose):
    scale: Vector3 = Field(default_factory=Vector3.one)

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Self:
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Expected a 4×4 matrix, got shape {m.shape}")
        sx = float(np.linalg.norm(m[:3, 0]))
        sy = float(np.linalg.norm(m[:3, 1]))
        sz = float(np.linalg.norm(m[:3, 2]))
        rot = np.zeros((3, 3), dtype=np.float64)
        if sx > 1e-12: rot[:, 0] = m[:3, 0] / sx
        if sy > 1e-12: rot[:, 1] = m[:3, 1] / sy
        if sz > 1e-12: rot[:, 2] = m[:3, 2] / sz
        return cls(
            position=Vector3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
            rotation=Quaternion.from_matrix(rot),
            scale=Vector3(sx, sy, sz),
        )

    def to_matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        r = self.rotation.to_matrix()
        m[:3, 0] = r[:, 0] * self.scale.x
        m[:3, 1] = r[:, 1] * self.scale.y
        m[:3, 2] = r[:, 2] * self.scale.z
        m[:3, 3] = self.position.to_numpy()
        return m

    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.to_matrix())

    def transform_normal(self, v: Vector3) -> Vector3:
        """Correct normal transformation under non-uniform scale."""
        rs_inv_T = np.linalg.inv(self.to_matrix()[:3, :3]).T
        return Vector3.from_sequence(rs_inv_T @ v.to_numpy()).normalized()

    def __matmul__(self, other: object) -> Self:
        if not isinstance(other, Transform):
            return NotImplemented
        return Transform.from_matrix(self.to_matrix() @ other.to_matrix())


def cosine_similarity(a: Vector3, b: Vector3) -> float:
    """
    Cosine similarity between two vectors, in [-1, 1].

    Raises ValueError if either vector is zero.
    """
    return float(np.cos(a.angle(b)))


def matrix_to_trs(m: np.ndarray) -> tuple[Vector3, Quaternion, Vector3]:
    """
    Decompose a 4×4 TRS matrix into (position, rotation, scale).

    Assumes no shear.  Scale is recovered from column lengths; rotation
    from the normalised upper-left 3×3.
    """
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected a 4×4 matrix, got shape {m.shape}")

    position = Vector3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))

    sx = float(np.linalg.norm(m[:3, 0]))
    sy = float(np.linalg.norm(m[:3, 1]))
    sz = float(np.linalg.norm(m[:3, 2]))
    scale = Vector3(sx, sy, sz)

    # Divide each column by its length to recover the pure rotation matrix
    rot = np.zeros((3, 3), dtype=np.float64)
    if sx > 1e-12: rot[:, 0] = m[:3, 0] / sx
    if sy > 1e-12: rot[:, 1] = m[:3, 1] / sy
    if sz > 1e-12: rot[:, 2] = m[:3, 2] / sz
    rotation = Quaternion.from_matrix(rot)

    return position, rotation, scale


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute the 2D convex hull of a set of points using the monotone chain
    algorithm.

    Args:
        points: (N, 2) array of 2D points.

    Returns:
        (K, 2) array of hull vertices in counter-clockwise order.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) array, got shape {pts.shape}")
    if pts.shape[0] <= 1:
        return pts.copy()

    # Sort lexicographically: primary key x, secondary key y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        """Signed area of triangle OAB (2D cross product)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    # Last point of each half duplicates the first of the other — remove them
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def point_to_ray_distance(point: Vector3, pose: Pose) -> float:
    """
    Shortest distance from a point to the ray defined by a Pose.

    The ray originates at pose.position and points along pose.forward (+Z).
    If the closest point on the infinite line lies behind the origin (t ≤ 0),
    the distance to the ray origin is returned instead.

    Args:
        point: The query point in world space.
        pose:  The pose defining ray origin and direction.

    Returns:
        Non-negative distance from point to the ray.
    """
    o = pose.position.to_numpy()
    d = pose.forward.to_numpy()
    p = point.to_numpy()

    v = p - o
    t = float(np.dot(v, d))

    if t <= 0.0:
        return float(np.linalg.norm(v))

    closest = o + t * d
    return float(np.linalg.norm(p - closest))
