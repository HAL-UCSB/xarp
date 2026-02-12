from xarp.entities import Element, DefaultAssets, ImageAsset
from xarp.express import SyncXR, AsyncXR, AsyncSimpleXR
from xarp.server import show_qrcode_link, run
from xarp.spatial import Pose


async def app(xarp: AsyncXR, params):
    asxr = AsyncSimpleXR(xarp.remote)
    model = await asxr.create_or_update_label("test", "foo", [0,1,1],[0,0,0],[1,1,1,1])
    print(model)


if __name__ == '__main__':
    show_qrcode_link()
    run(app)
