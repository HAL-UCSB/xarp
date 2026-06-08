import unittest


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


if __name__ == "__main__":
    unittest.main()
