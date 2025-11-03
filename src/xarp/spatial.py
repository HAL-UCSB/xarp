'''
Representation: a single immutable Transform object with fields:
  - rotation: unit quaternion (w, x, y, z), right-handed, det(R) = +1
  - position: 3-vector (tx, ty, tz)
  - scale: positive uniform scalar s > 0

Conventions:
  - Right-handed coordinates.
  - Quaternion is (w, x, y, z).
  - Applying the transform to a point x computes: y = s * R @ x + t.
  - Composition uses column-vector, post-application semantics:
      (T_self ∘ T_other)(x) = T_self(T_other(x)).
    That is, `T = T_self @ T_other` means 'apply T_other, then T_self'.
'''

import math
from typing import Iterable, Tuple, Union, Any, List

import numpy as np
from pydantic import BaseModel, field_validator, model_validator, ConfigDict, field_serializer

FloatArrayLike = Union[Iterable[float], np.ndarray]


# ---- Quaternion and rotation helpers -----------------------------------------
def _as_np1d(x: FloatArrayLike, length: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != length:
        raise ValueError(f'{name} must have length {length}, got shape {arr.shape}.')
    return arr


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError('Quaternion norm must be positive and finite.')
    qn = q / n
    # Enforce a canonical sign: w >= 0 for uniqueness
    if qn[0] < 0:
        qn = -qn
    return qn


def _quat_to_matrix(qwxyz: np.ndarray) -> np.ndarray:
    '''Quaternion (w, x, y, z) to 3x3 rotation matrix, right-handed.'''
    w, x, y, z = qwxyz
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )
    return R


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    '''3x3 rotation matrix to quaternion (w, x, y, z), right-handed, det(R)=+1.'''
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError('Rotation matrix must be 3x3.')
    # Orthonormalize defensively via SVD to avoid drift.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Choose the largest diagonal for numerical stability.
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = _quat_normalize(np.array([w, x, y, z], dtype=float))
    return q


def _euler_xyz_to_quat(angles: FloatArrayLike, degrees: bool = False) -> np.ndarray:
    '''
    Convert intrinsic XYZ Euler angles to a quaternion (w, x, y, z).
    Intrinsic rotations about body axes, applied in order X, then Y, then Z.
    Right-handed positive angles, standard right-hand rule.
    '''
    ax, ay, az = _as_np1d(angles, 3, 'angles')
    if degrees:
        ax, ay, az = np.deg2rad([ax, ay, az])

    cx, sx = math.cos(ax / 2.0), math.sin(ax / 2.0)
    cy, sy = math.cos(ay / 2.0), math.sin(ay / 2.0)
    cz, sz = math.cos(az / 2.0), math.sin(az / 2.0)

    # q = qz * qy * qx for intrinsic XYZ
    qw = cz * cy * cx + sz * sy * sx
    qx = cz * cy * sx - sz * sy * cx
    qy = cz * sy * cx + sz * cy * sx
    qz = sz * cy * cx - cz * sy * sx

    return _quat_normalize(np.array([qw, qx, qy, qz], dtype=float))


# ---- Similarity Transform -----------------------------------------------------
class Transform(BaseModel):
    '''
    Right-handed Sim(3) similarity transform with uniform scale.

    Fields
    ------
    rotation : array-like of length 4
        Unit quaternion (w, x, y, z), right-handed, det(R) = +1.
    position : array-like of length 3
        position vector (tx, ty, tz).
    scale : float
        Positive uniform scale s > 0.

    Semantics
    ---------
    Applying the transform to a point x computes y = s * R @ x + t.
    Composition T = A @ B satisfies T(x) = A(B(x)).

    Immutability
    ------------
    Instances are immutable; all mutating operations return new instances.
    '''

    rotation: Any  # will be validated to np.ndarray shape (4,)
    position: Any  # will be validated to np.ndarray shape (3,)
    scale: float = 1.0

    # Pydantic configuration
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ---- Validators  ----------------------------------------------------------

    @field_validator('rotation')
    @classmethod
    def _val_rotation(cls, v: Any) -> np.ndarray:
        q = _as_np1d(v, 4, 'rotation')
        return _quat_normalize(q)

    @field_validator('position')
    @classmethod
    def _val_position(cls, v: Any) -> np.ndarray:
        return _as_np1d(v, 3, 'position')

    @model_validator(mode='after')
    def _val_scale(self) -> 'Transform':
        if not (isinstance(self.scale, (int, float)) and np.isfinite(self.scale) and self.scale > 0.0):
            raise ValueError('scale must be a positive finite float.')
        return self

    # ---- Constructors ---------------------------------------------------------
    @classmethod
    def identity(cls) -> 'Transform':
        return cls(rotation=np.array([1.0, 0.0, 0.0, 0.0]), position=np.zeros(3), scale=1.0)

    @classmethod
    def from_position(cls, *position):
        return cls(rotation=np.array([1, 0, 0, 0]), position=np.array(position), scale=1.0)

    @classmethod
    def from_quaternion(cls, q: FloatArrayLike, t: FloatArrayLike = (0.0, 0.0, 0.0), s: float = 1.0) -> 'Transform':
        return cls(rotation=_quat_normalize(_as_np1d(q, 4, 'q')), position=_as_np1d(t, 3, 't'), scale=float(s))

    @classmethod
    def from_euler_xyz(cls, angles: FloatArrayLike, t: FloatArrayLike = (0.0, 0.0, 0.0), s: float = 1.0,
                       degrees: bool = False) -> 'Transform':
        q = _euler_xyz_to_quat(angles, degrees=degrees)
        return cls(rotation=q, position=_as_np1d(t, 3, 't'), scale=float(s))

    @classmethod
    def from_matrix(cls, M: FloatArrayLike) -> 'Transform':
        '''
        Construct from a 4×4 homogeneous matrix with positive uniform scale.
        Enforces right-handed rotation with det(R) = +1. Raises if the
        linear part has non-uniform scale or reflection.
        '''
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError('Matrix must be 4x4.')
        A = M[:3, :3]
        t = M[:3, 3]

        # Extract uniform scale as the mean of column norms, then verify uniformity.
        col_norms = np.linalg.norm(A, axis=0)
        s = float(np.mean(col_norms))
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError('Matrix scale must be positive and finite.')
        A_norm = A / s

        # Orthonormalize and ensure det(R) = +1.
        U, _, Vt = np.linalg.svd(A_norm)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Verify uniformity within tolerance.
        tol = 1e-6
        if not np.allclose(A_norm, R, atol=tol):
            # If it looks like anisotropic scale or shear, reject.
            raise ValueError('Input matrix contains shear or non-uniform scale; expected uniform scale only.')

        q = _matrix_to_quat(R)
        return cls(rotation=q, position=t, scale=s)

    # ---- Basic exports --------------------------------------------------------
    def rotation_matrix(self) -> np.ndarray:
        return _quat_to_matrix(self.rotation)

    def matrix(self) -> np.ndarray:
        '''Return the 4×4 homogeneous matrix.'''
        R = self.rotation_matrix()
        M = np.eye(4, dtype=float)
        M[:3, :3] = self.scale * R
        M[:3, 3] = self.position
        return M

    # ---- Apply to data -------------------------------------------------------
    def apply(self, points: FloatArrayLike) -> np.ndarray:
        '''
        Apply to 3D points of shape (..., 3).
        y = s * R @ x + t
        '''
        pts = np.asarray(points, dtype=float)
        if pts.shape[-1] != 3:
            raise ValueError(f'points must have shape (..., 3); got {pts.shape}.')
        R = self.rotation_matrix()
        y = self.scale * (pts @ R.T) + self.position
        return y

    def apply_directions(self, dirs: FloatArrayLike, normalize: bool = False) -> np.ndarray:
        '''
        Apply to direction vectors (no translation): y = s * R @ v.
        For uniform scale, directions scale uniformly; optionally renormalize.
        '''
        v = np.asarray(dirs, dtype=float)
        if v.shape[-1] != 3:
            raise ValueError(f'dirs must have shape (..., 3); got {v.shape}.')
        R = self.rotation_matrix()
        y = self.scale * (v @ R.T)
        if normalize:
            n = np.linalg.norm(y, axis=-1, keepdims=True)
            n = np.where(n == 0.0, 1.0, n)
            y = y / n
        return y

    def apply_normals(self, normals: FloatArrayLike, renormalize: bool = True) -> np.ndarray:
        '''
        Transform surface normals. For uniform scale:
          n' ∝ R @ n
        Uniform scaling does not change normal direction; we optionally renormalize.
        '''
        n = np.asarray(normals, dtype=float)
        if n.shape[-1] != 3:
            raise ValueError(f'normals must have shape (..., 3); got {n.shape}.')
        R = self.rotation_matrix()
        y = n @ R.T
        if renormalize:
            m = np.linalg.norm(y, axis=-1, keepdims=True)
            m = np.where(m == 0.0, 1.0, m)
            y = y / m
        return y

    # ---- Algebra --------------------------------------------------------------
    def inverse(self) -> 'Transform':
        '''
        Inverse transform:
          R_inv = R^T, s_inv = 1/s, t_inv = -(1/s) * R^T @ t
        '''
        R = self.rotation_matrix()
        Rt = R.T
        s_inv = 1.0 / self.scale
        t_inv = -(s_inv * (Rt @ self.position))
        q_inv = self.conjugate_quaternion()
        return Transform(rotation=q_inv, position=t_inv, scale=s_inv)

    def conjugate_quaternion(self) -> np.ndarray:
        '''Return quaternion conjugate (w, -x, -y, -z) for unit quaternion.'''
        w, x, y, z = self.rotation
        return np.array([w, -x, -y, -z], dtype=float)

    def __matmul__(self, other: 'Transform') -> 'Transform':
        '''
        Composition via the @ operator:
        (self @ other)(x) = self(self_other(x)).

        Given self: (s1, R1, t1) and other: (s2, R2, t2),
        T = (s1*s2, R1 @ R2, s1 * R1 @ t2 + t1)
        '''
        if not isinstance(other, Transform):
            return NotImplemented
        s = self.scale * other.scale
        # Compose rotation by matrix then convert back to quaternion
        R = self.rotation_matrix() @ other.rotation_matrix()
        q = _matrix_to_quat(R)
        t = self.scale * (self.rotation_matrix() @ other.position) + self.position
        return Transform(rotation=q, position=t, scale=s)

    # Alias for composition to be explicit
    def compose(self, other: 'Transform') -> 'Transform':
        return self @ other

    # ---- Utilities ------------------------------------------------------------
    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, float]:
        '''Return (rotation_quat_wxyz, position_xyz, scale).'''
        return self.rotation.copy(), self.position.copy(), float(self.scale)

    def to_jsonable(self) -> dict:
        '''Plain-Python structure suitable for JSON serialization.'''
        return {
            'rotation': self.rotation.tolist(),
            'position': self.position.tolist(),
            'scale': float(self.scale),
        }

    # Pretty representation
    def __repr__(self) -> str:  # pragma: no cover
        r = np.array2string(self.rotation, precision=6, separator=', ')
        t = np.array2string(self.position, precision=6, separator=', ')
        return f'Transform(rotation={r}, position={t}, scale={self.scale:.6f})'

    # ---- Serializers ----------------------------------------------------------
    @field_serializer('position')
    def _serialize_position(self, position: np.ndarray) -> List[float]:
        return self.position.tolist()

    @field_serializer('rotation')
    def _serialize_rotation(self, rotation: np.ndarray) -> List[float]:
        return self.rotation.tolist()


# ---- Tests ------------------------------------------
if __name__ == '__main__':
    T_id = Transform.identity()
    assert np.allclose(T_id.matrix(), np.eye(4))

    T1 = Transform.from_euler_xyz([0.1, 0.2, 0.3], t=[1, 2, 3], s=2.0)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]])
    out = T1.apply(pts)

    T2 = Transform.from_quaternion([1, 0, 0, 0], [0.5, 0, 0], 0.5)
    T12 = T1 @ T2
    out_comp = T12.apply(pts)
    out_seq = T1.apply(T2.apply(pts))
    assert np.allclose(out_comp, out_seq)

    # Round-trip through matrix
    M = T1.matrix()
    T1_rt = Transform.from_matrix(M)
    assert np.allclose(T1_rt.matrix(), M, atol=1e-9)
