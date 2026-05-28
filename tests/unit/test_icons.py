import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageFont

from xarp.entities import ImageAsset
from xarp.icons import (
    _material_symbol_codepoints,
    _material_symbol_font_bytes,
    _normalize_rgba,
    material_symbol_asset,
    material_symbol_image,
)


class TestMaterialSymbols(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_font = ImageFont.load_default()

    def setUp(self):
        _material_symbol_codepoints.cache_clear()
        _material_symbol_font_bytes.cache_clear()

    def tearDown(self):
        _material_symbol_codepoints.cache_clear()
        _material_symbol_font_bytes.cache_clear()

    def _mock_response(self, text: str = "", content: bytes = b"") -> Mock:
        response = Mock()
        response.text = text
        response.content = content
        response.raise_for_status = Mock()
        return response

    def test_downloads_codepoints_and_font_once(self):
        responses = [
            self._mock_response(text="home e88a\ncancel e5c9\n"),
            self._mock_response(content=b"font-bytes"),
        ]

        with (
            patch("xarp.icons.requests.get", side_effect=responses) as get,
            patch("xarp.icons.ImageFont.truetype", return_value=self.default_font),
        ):
            first = material_symbol_image("home", size=32)
            second = material_symbol_image("cancel", size=32)

        self.assertIsInstance(first, Image.Image)
        self.assertIsInstance(second, Image.Image)
        self.assertEqual(get.call_count, 2)

    def test_unknown_icon_raises_value_error(self):
        with patch(
            "xarp.icons.requests.get",
            return_value=self._mock_response(text="home e88a\n"),
        ):
            with self.assertRaisesRegex(ValueError, "Icon 'missing' not found"):
                material_symbol_image("missing", size=32)

    def test_tuple_colors_convert_to_rgba_255_values(self):
        self.assertEqual(
            _normalize_rgba((0.25, 0.5, 1.0, 0.75)),
            (63, 127, 255, 191),
        )
        self.assertEqual(_normalize_rgba((1.0, 0.0, 0.5)), (255, 0, 127, 255))

    def test_returns_rgba_image_with_requested_size(self):
        responses = [
            self._mock_response(text="home e88a\n"),
            self._mock_response(content=b"font-bytes"),
        ]

        with (
            patch("xarp.icons.requests.get", side_effect=responses),
            patch("xarp.icons.ImageFont.truetype", return_value=self.default_font),
        ):
            image = material_symbol_image("home", size=48)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (48, 48))

    def test_material_symbol_asset_returns_image_asset(self):
        responses = [
            self._mock_response(text="home e88a\n"),
            self._mock_response(content=b"font-bytes"),
        ]

        with (
            patch("xarp.icons.requests.get", side_effect=responses),
            patch("xarp.icons.ImageFont.truetype", return_value=self.default_font),
        ):
            asset = material_symbol_asset("home", size=32, asset_key="home-icon")

        self.assertIsInstance(asset, ImageAsset)
        self.assertEqual(asset.asset_key, "home-icon")
        self.assertIsNotNone(asset.raw)


if __name__ == "__main__":
    unittest.main()
