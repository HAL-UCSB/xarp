import unittest
from io import BytesIO

from PIL import Image

from xarp.entities import ImageAsset, MIMEType


class TestAgentTools(unittest.TestCase):

    def test_sync_simple_xr_methods_can_be_smolagents_tools(self):
        try:
            import smolagents
            from xarp.agents import _get_public_methods
            from xarp.express import SyncSimpleXR
        except ImportError as exc:
            raise unittest.SkipTest("agent dependencies are not installed") from exc

        xr = SyncSimpleXR.__new__(SyncSimpleXR)

        for name, method in _get_public_methods(xr):
            with self.subTest(name=name):
                smolagents.tool(method)

    def test_async_simple_xr_methods_can_be_fastmcp_tools(self):
        try:
            import fastmcp
            from xarp.agents import _get_public_methods
            from xarp.express import AsyncSimpleXR
        except ImportError as exc:
            raise unittest.SkipTest("agent dependencies are not installed") from exc

        xr = AsyncSimpleXR.__new__(AsyncSimpleXR)
        mcp = fastmcp.FastMCP("smoke")

        for name, method in _get_public_methods(xr):
            with self.subTest(name=name):
                mcp.tool(method)

    def test_image_asset_becomes_mcp_image_content(self):
        try:
            from xarp.express import _asset_to_mcp_image
        except ImportError as exc:
            raise unittest.SkipTest("agent dependencies are not installed") from exc

        image = Image.new("RGB", (2, 2), color=(255, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        raw = buffer.getvalue()

        asset = ImageAsset(mime_type=MIMEType.JPEG, raw=raw)
        content = _asset_to_mcp_image(asset).to_image_content()

        self.assertEqual(content.type, "image")
        self.assertEqual(content.mimeType, MIMEType.JPEG)
        self.assertEqual(content.data, asset.to_base64())


if __name__ == "__main__":
    unittest.main()
