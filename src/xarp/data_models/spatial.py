import math
from typing import Any, Optional
from typing import Iterable, Tuple, Union, List

import numpy as np
from pydantic import BaseModel, Field
from pydantic import RootModel
from pydantic import field_validator, model_validator, ConfigDict, field_serializer


class Vector3(RootModel[list[float]]):

    @property
    def x(self) -> float:
        return self.root[0]

    @property
    def y(self) -> float:
        return self.root[1]

    @property
    def z(self) -> float:
        return self.root[2]

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Vector3':
        return cls([float(x), float(y), float(z)])

    @staticmethod
    def zero() -> 'Vector3':
        return Vector3([0.0, 0.0, 0.0])

    @staticmethod
    def one() -> 'Vector3':
        return Vector3([1.0, 1.0, 1.0])

    def to_numpy(self) -> np.ndarray:
        return np.array(self.root, dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.root))

    def normalized(self) -> 'Vector3':
        arr = self.to_numpy()
        n = np.linalg.norm(arr)
        if n == 0.0:
            raise ValueError('Cannot normalize zero vector')
        return Vector3((arr / n).tolist())

    def __add__(self, other: 'Vector3') -> 'Vector3':
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3([
            self.root[0] + other.root[0],
            self.root[1] + other.root[1],
            self.root[2] + other.root[2],
        ])

    def __radd__(self, other: 'Vector3') -> 'Vector3':
        return self.__add__(other)

    def __mul__(self, scalar: float) -> 'Vector3':
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector3([
            self.root[0] * scalar,
            self.root[1] * scalar,
            self.root[2] * scalar,
        ])

    def __rmul__(self, scalar: float) -> 'Vector3':
        return self.__mul__(scalar)


class Quaternion(RootModel[list[float]]):
    @classmethod
    def from_wxyz(cls, w: float, x: float, y: float, z: float) -> 'Quaternion':
        return cls([float(w), float(x), float(y), float(z)])

    @property
    def w(self) -> float:
        return self.root[0]

    @property
    def x(self) -> float:
        return self.root[1]

    @property
    def y(self) -> float:
        return self.root[2]

    @property
    def z(self) -> float:
        return self.root[3]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.root, dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.root))

    def normalized(self) -> 'Quaternion':
        arr = self.to_numpy()
        n = np.linalg.norm(arr)
        if n == 0.0:
            raise ValueError('Cannot normalize zero quaternion')
        return Quaternion((arr / n).tolist())

    @staticmethod
    def zero() -> 'Quaternion':
        return Quaternion([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def identity() -> 'Quaternion':
        return Quaternion([1.0, 0.0, 0.0, 0.0])


class Pose(BaseModel):
    position: Vector3 = Field(default_factory=Vector3.zero)
    rotation: Quaternion = Field(default_factory=Quaternion.identity)

    def to_matrix(self) -> np.ndarray:
        # Scale matrix
        S = np.eye(4)
        S[0, 0] = 1
        S[1, 1] = 1
        S[2, 2] = 1

        # Rotation matrix
        R = self.rotation.to_rotation_matrix()

        # Translation matrix
        T = np.eye(4)
        T[:3, 3] = np.asarray(self.position, dtype=float)

        # Combine (T * R * S)
        return T @ R @ S


class Transform(Pose):
    parent: Optional['Transform'] = None
    scale: Vector3 = Field(default_factory=Vector3.one)

    def to_matrix(self) -> np.ndarray:
        # Scale matrix
        S = np.eye(4)
        S[0, 0] = self.scale.x
        S[1, 1] = self.scale.y
        S[2, 2] = self.scale.z

        # Rotation matrix
        R = self.rotation.to_rotation_matrix()

        # Translation matrix
        T = np.eye(4)
        T[:3, 3] = np.asarray(self.position, dtype=float)

        # Combine (T * R * S)
        return T @ R @ S

    def world_matrix(self) -> np.ndarray:
        M = self.to_matrix()
        if self.parent is not None:
            return self.parent.world_matrix() @ M
        return M


def distance(a: Vector3, b: Vector3):
    return np.linalg.norm(a.to_numpy() - b.to_numpy())


def cosine_similarity(a: Vector3, b: Vector3):
    a = a.to_numpy()
    b = b.to_numpy()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def centroid(transforms: Iterable[Transform]):
    positions = [t.position for t in transforms]
    return np.array(positions).mean(axis=0)


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