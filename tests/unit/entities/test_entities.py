import base64
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image
from pydantic import ValidationError

from xarp.entities import (
    Asset,
    DefaultAssets,
    Element,
    GLBAsset,
    ImageAsset,
    MIMEType,
    TextAsset,
)
from xarp.spatial import Transform


# ---------------------------------------------------------------------------
# Remote fixture URLs — fetched once per TestCase via setUpClass
# ---------------------------------------------------------------------------

OGG_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg"
GLB_URL = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"


def fetch(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; xarp-tests/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


# ---------------------------------------------------------------------------
# MIMEType
# ---------------------------------------------------------------------------

class TestMIMEType(unittest.TestCase):

    def test_from_extension_with_dot(self):
        self.assertEqual(MIMEType.from_extension(".png"), MIMEType.PNG)

    def test_from_extension_without_dot(self):
        self.assertEqual(MIMEType.from_extension("jpg"), MIMEType.JPEG)

    def test_from_extension_ogg_fallback(self):
        self.assertEqual(MIMEType.from_extension(".ogg"), MIMEType.OGG)

    def test_from_extension_glb_fallback(self):
        self.assertEqual(MIMEType.from_extension(".glb"), MIMEType.GLB)

    def test_from_extension_txt(self):
        self.assertEqual(MIMEType.from_extension("txt"), MIMEType.TXT)


# ---------------------------------------------------------------------------
# Asset (base)
# ---------------------------------------------------------------------------

class TestAsset(unittest.TestCase):

    def test_construction_with_raw(self):
        a = Asset(asset_key="k", mime_type=MIMEType.TXT, raw=b"hello")
        self.assertEqual(a.raw, b"hello")

    def test_obj_returns_raw_by_default(self):
        a = Asset(raw=b"hello")
        self.assertEqual(a.obj, b"hello")

    def test_obj_raises_when_no_raw(self):
        a = Asset()
        with self.assertRaises(RuntimeError):
            _ = a.obj

    def test_obj_is_not_cached(self):
        # Each call to obj deserialises fresh — the two objects are equal but not identical
        a = Asset(raw=b"hello")
        self.assertEqual(a.obj, a.obj)

    def test_from_obj_populates_raw_immediately(self):
        a = Asset.from_obj(b"data", mime_type=MIMEType.TXT)
        self.assertIsNotNone(a.raw)

    def test_serialize_returns_raw(self):
        a = Asset(asset_key="k", mime_type=MIMEType.TXT, raw=b"hi")
        d = a.serialize()
        self.assertEqual(d["raw"], b"hi")
        self.assertEqual(d["asset_key"], "k")

    def test_serialize_raw_none_when_empty(self):
        a = Asset(asset_key="k")
        d = a.serialize()
        self.assertIsNone(d["raw"])

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Asset(raw=b"x", unknown=True)


# ---------------------------------------------------------------------------
# DefaultAssets
# ---------------------------------------------------------------------------

class TestDefaultAssets(unittest.TestCase):

    def test_each_call_returns_fresh_instance(self):
        a = DefaultAssets.sphere()
        b = DefaultAssets.sphere()
        self.assertIsNot(a, b)

    def test_sphere_has_raw(self):
        self.assertIsNotNone(DefaultAssets.sphere().raw)

    def test_all_have_xarp_default_mime(self):
        factories = [
            DefaultAssets.sphere, DefaultAssets.cube, DefaultAssets.capsule,
            DefaultAssets.cylinder, DefaultAssets.plane, DefaultAssets.quad,
            DefaultAssets.axes,
        ]
        for f in factories:
            with self.subTest(name=f.__name__):
                self.assertEqual(f().mime_type, MIMEType.XARP_DEFAULT)

    def test_mutation_does_not_affect_next_instance(self):
        a = DefaultAssets.cube()
        # validate_assignment is True so this should work on this instance
        a.raw = b"mutated"
        b = DefaultAssets.cube()
        self.assertNotEqual(b.raw, b"mutated")


# ---------------------------------------------------------------------------
# ImageAsset
# ---------------------------------------------------------------------------

def _make_png_bytes(width=4, height=4) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestImageAsset(unittest.TestCase):

    def setUp(self):
        self.png_bytes = _make_png_bytes()

    def test_construction_with_raw(self):
        a = ImageAsset(raw=self.png_bytes)
        self.assertEqual(a.mime_type, MIMEType.PNG)

    def test_obj_returns_pil_image(self):
        a = ImageAsset(raw=self.png_bytes)
        img = a.obj
        self.assertIsInstance(img, Image.Image)

    def test_obj_round_trips_dimensions(self):
        a = ImageAsset(raw=self.png_bytes)
        self.assertEqual(a.obj.size, (4, 4))

    def test_from_obj_encodes_raw(self):
        img = Image.new("RGB", (2, 2), color=(0, 255, 0))
        a = ImageAsset.from_obj(img)
        self.assertIsNotNone(a.raw)

    def test_from_obj_round_trip(self):
        img = Image.new("RGB", (8, 8), color=(0, 0, 255))
        a = ImageAsset.from_obj(img)
        recovered = a.obj
        self.assertEqual(recovered.size, img.size)

    def test_to_base64_is_valid(self):
        a = ImageAsset(raw=self.png_bytes)
        encoded = a.to_base64()
        decoded = base64.b64decode(encoded)
        self.assertEqual(decoded, self.png_bytes)

    def test_default_mime_is_png(self):
        a = ImageAsset(raw=self.png_bytes)
        self.assertEqual(a.mime_type, MIMEType.PNG)

    def test_jpeg_mime_accepted(self):
        img = Image.new("RGB", (2, 2))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        a = ImageAsset(mime_type=MIMEType.JPEG, raw=buf.getvalue())
        self.assertEqual(a.mime_type, MIMEType.JPEG)

    def test_wrong_mime_rejected(self):
        with self.assertRaises(ValidationError):
            ImageAsset(mime_type=MIMEType.TXT, raw=self.png_bytes)


# ---------------------------------------------------------------------------
# TextAsset
# ---------------------------------------------------------------------------

class TestTextAsset(unittest.TestCase):

    def test_construction_with_raw(self):
        a = TextAsset(raw=b"hello world")
        self.assertEqual(a.obj, "hello world")

    def test_from_obj_encodes_utf8(self):
        a = TextAsset.from_obj("héllo")
        self.assertEqual(a.raw, "héllo".encode("utf-8"))

    def test_obj_decodes_utf8(self):
        a = TextAsset(raw="héllo".encode("utf-8"))
        self.assertEqual(a.obj, "héllo")

    def test_from_obj_round_trip(self):
        text = "the quick brown fox"
        a = TextAsset.from_obj(text)
        self.assertEqual(a.obj, text)

    def test_mime_is_always_txt(self):
        a = TextAsset(raw=b"x")
        self.assertEqual(a.mime_type, MIMEType.TXT)

    def test_wrong_mime_rejected(self):
        with self.assertRaises(ValidationError):
            TextAsset(mime_type=MIMEType.PNG, raw=b"x")


# ---------------------------------------------------------------------------
# GLBAsset — uses Duck.glb from KhronosGroup via HTTP
# ---------------------------------------------------------------------------

class TestGLBAsset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.glb_bytes = fetch(GLB_URL)

    def test_construction_with_raw(self):
        a = GLBAsset(raw=self.glb_bytes)
        self.assertIsNotNone(a.raw)

    def test_mime_is_glb(self):
        a = GLBAsset(raw=self.glb_bytes)
        self.assertEqual(a.mime_type, MIMEType.GLB)

    def test_obj_returns_trimesh(self):
        import trimesh
        a = GLBAsset(raw=self.glb_bytes)
        mesh = a.obj
        self.assertIsInstance(mesh, trimesh.Trimesh)

    def test_from_obj_round_trip(self):
        import trimesh
        original = trimesh.load(BytesIO(self.glb_bytes), file_type="glb")
        a = GLBAsset.from_obj(original)
        self.assertIsNotNone(a.raw)
        recovered = a.obj
        self.assertIsInstance(recovered, trimesh.Trimesh)

    def test_wrong_mime_rejected(self):
        with self.assertRaises(ValidationError):
            GLBAsset(mime_type=MIMEType.PNG, raw=self.glb_bytes)


# ---------------------------------------------------------------------------
# Asset[bytes] used as a generic audio container (OGG via HTTP)
# ---------------------------------------------------------------------------

class TestAudioAsset(unittest.TestCase):
    """
    Audio types (OGG, MP3, WAV) have no typed subclass yet — they are
    stored as raw bytes in a plain Asset. These tests use the Wikimedia
    OGG fixture to verify the base Asset round-trips binary data faithfully.
    """

    @classmethod
    def setUpClass(cls):
        cls.ogg_bytes = fetch(OGG_URL)

    def test_raw_preserved_exactly(self):
        a = Asset(mime_type=MIMEType.OGG, raw=self.ogg_bytes)
        self.assertEqual(a.raw, self.ogg_bytes)

    def test_obj_returns_same_bytes(self):
        a = Asset(mime_type=MIMEType.OGG, raw=self.ogg_bytes)
        self.assertEqual(a.obj, self.ogg_bytes)

    def test_from_obj_round_trip(self):
        a = Asset.from_obj(self.ogg_bytes, mime_type=MIMEType.OGG, asset_key="example")
        self.assertEqual(a.raw, self.ogg_bytes)
        self.assertEqual(a.asset_key, "example")

    def test_serialize_preserves_raw(self):
        a = Asset(mime_type=MIMEType.OGG, raw=self.ogg_bytes)
        d = a.serialize()
        self.assertEqual(d["raw"], self.ogg_bytes)


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------

class TestElement(unittest.TestCase):

    def test_default_construction(self):
        e = Element()
        self.assertEqual(e.key, "")
        self.assertTrue(e.active)
        self.assertIsNone(e.asset)
        self.assertIsInstance(e.transform, Transform)

    def test_element_without_asset_is_valid(self):
        e = Element(key="anchor")
        self.assertIsNone(e.asset)

    def test_element_with_valid_asset(self):
        a = TextAsset.from_obj("hello")
        e = Element(key="label", asset=a)
        self.assertIsInstance(e.asset, TextAsset)
        self.assertEqual(e.asset.obj, "hello")

    def test_element_with_empty_asset_accepted(self):
        a = Asset()  # raw=None
        Element(asset=a)

    def test_element_with_default_asset_accepted(self):
        e = Element(asset=DefaultAssets.sphere())
        self.assertIsNotNone(e.asset)

    def test_color_none_by_default(self):
        e = Element()
        self.assertIsNone(e.color)

    def test_color_rgba_accepted(self):
        e = Element(color=(1.0, 0.5, 0.0, 1.0))
        self.assertEqual(e.color, (1.0, 0.5, 0.0, 1.0))

    def test_transform_is_independent_per_instance(self):
        e1 = Element()
        e2 = Element()
        self.assertIsNot(e1.transform, e2.transform)

    def test_active_flag(self):
        e = Element(active=False)
        self.assertFalse(e.active)

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Element(unknown_field=True)

    def test_play_and_time_accepted(self):
        e = Element(play=True, time=3.5)
        self.assertTrue(e.play)
        self.assertAlmostEqual(e.time, 3.5)

    def test_play_and_time_default_none(self):
        e = Element()
        self.assertIsNone(e.play)
        self.assertIsNone(e.time)


if __name__ == "__main__":
    unittest.main()