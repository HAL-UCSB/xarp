import asyncio
import threading
from threading import Thread
from typing import Any, Iterator
from typing import AsyncGenerator

from xarp.commands import Bundle, ResponseMode
from xarp.commands.entities import (
    ListAssetsCommand,
    DestroyElementCommand,
    DestroyAssetCommand, ListElementsCommand, CreateOrUpdateAssetsCommand, CreateOrUpdateElementCommand,
)
from xarp.commands.info import InfoCommand
from xarp.commands.sensing import (
    ImageCommand,
    EyeCommand,
    HeadCommand,
    HandsCommand,
    DepthCommand,
    VirtualImageCommand,
)
from xarp.commands.ui import WriteCommand, SayCommand, ReadCommand, PassthroughCommand
from xarp.data_models import DeviceInfo, Hands
from xarp.entities import ImageAsset, Asset, Element
from xarp.remote import RemoteXRClient
from xarp.spatial import Pose


class AsyncXR:
    """Async convenience wrapper around a :class:`~xarp.RemoteXRClient` that exposes common XR operations"""

    def __init__(self, remote: RemoteXRClient):
        self.remote = remote

    async def _execute_none(self, *cmds: Any) -> None:
        await self.remote.execute(Bundle(cmds=list(cmds), mode=ResponseMode.NONE))

    async def _execute_single(self, *cmds: Any) -> Any:
        resp = await self.remote.execute(Bundle(cmds=list(cmds), mode=ResponseMode.SINGLE))
        return resp.value[0] if len(cmds) == 1 else resp.value

    # ---- INFO ----

    async def info(self) -> DeviceInfo:
        """Returns remote system information.

        Returns:
            DeviceInfo describing runtime/device capabilities and configuration.
        """
        return await self._execute_single(InfoCommand())

    # ---- UI ----

    async def write(self, text: str, title: str | None = None) -> None:
        """Displays a text message.

        Args:
            text: Message content to display.
            title: Optional title displayed alongside the message.

        Returns:
            None.
        """
        await self._execute_none(WriteCommand(text=text, title=title))

    async def say(self, text: str, title: str | None = None) -> None:
        """Displays a text message and triggers synthesized speech. Resolves when speech playback completes.

        Args:
            text: Message content to display and speak.
            title: Optional title displayed alongside the message. (Not spoken)

        Returns:
            None.
        """
        await self.remote.execute(
            Bundle(cmds=[SayCommand(text=text, title=title)], mode=ResponseMode.SINGLE)
        )

    async def read(self, text: str, title: str | None = None) -> str:
        """Prompts the user for text input and returns it.

        Args:
            text: Prompt content presented to the user.
            title: Optional title displayed alongside the prompt.

        Returns:
            The user's entered text.
        """
        return await self._execute_single(ReadCommand(text=text, title=title))

    async def passthrough(self, transparency: float) -> None:
        """Sets passthrough transparency.
        Args:
            transparency: Ranges from 0.0 to 1.0. 0.0 is fully virtual, 1.0 is full passthrough.

        Returns:
            None.
        """
        await self._execute_none(PassthroughCommand(transparency=transparency))

    # ---- SENSING (SINGLE) ----

    async def image(self) -> ImageAsset:
        """Captures one RGB image of the physical environment.
        Returns:
            ImageResource containing an RGB image from the user's point of view.
        """
        return await self._execute_single(ImageCommand())

    async def virtual_image(self) -> ImageAsset:
        """Captures one RGBA image of the virtual environment.
        Returns:
            ImageResource containing an RGBA render from the user's point of view.
        """
        return await self._execute_single(VirtualImageCommand())

    async def depth(self) -> ImageAsset:
        """Captures one depth frame of the physical environment.
        Returns:
            ImageResource containing a depth image from the user's point of view.
        """
        return await self._execute_single(DepthCommand())

    async def eye(self) -> Pose:
        """Returns the pose of the main camera (camera extrinsics).
        Returns:
            Pose of the device's main camera, typically close to the user's eye pose.
        """
        return await self._execute_single(EyeCommand())

    async def head(self) -> Pose:
        """Returns the pose of the XR device (typically the user's head pose).
        Returns:
            Pose of the headset in the runtime coordinate frame.
        """
        return await self._execute_single(HeadCommand())

    async def hands(self) -> Hands:
        """Returns tracked hand joint poses.
        Returns:
            Hands payload. ``left`` and ``right`` are None when the corresponding hand
            is not currently tracked.
        """
        return await self._execute_single(HandsCommand())

    # ---- SENSING (STREAM) ----

    async def sense(
            self,
            *,
            image: bool = False,
            virtual_image: bool = False,
            depth: bool = False,
            eye: bool = False,
            head: bool = False,
            hands: bool = False,
            rt: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streams selected sensing modalities continuously. Values are produced later by iterating the returned async generator. Cleanup requires an explict call to the aclose to the returned async generator.
        Args:
            image: If True, include RGB physical-camera frames under key ``"image"``.
            virtual_image: If True, include RGBA virtual-render frames under key
                ``"virtual_image"``.
            depth: If True, include depth frames under key ``"depth"``.
            eye: If True, include camera pose under key ``"eye"``.
            head: If True, include headset pose under key ``"head"``.
            hands: If True, include hand-tracking payload under key ``"hands"``.
            rt: If True, yields the latest available sample at read time (dropping
                intermediate frames). If False, yields samples sequentially even if
                delayed.

        Yields:
            Dictionaries mapping enabled modality keys to their corresponding values.
        """
        keys: list[str] = []
        cmds: list[Any] = []

        if image:
            keys.append("image")
            cmds.append(ImageCommand())
        if virtual_image:
            keys.append("virtual_image")
            cmds.append(VirtualImageCommand())
        if depth:
            keys.append("depth")
            cmds.append(DepthCommand())
        if eye:
            keys.append("eye")
            cmds.append(EyeCommand())
        if head:
            keys.append("head")
            cmds.append(HeadCommand())
        if hands:
            keys.append("hands")
            cmds.append(HandsCommand())

        if not cmds:
            return

        stream = await self.remote.execute(Bundle(cmds=cmds, mode=ResponseMode.STREAM, rt=rt))
        try:
            async for item in stream:
                yield dict(zip(keys, item.value, strict=True))
        finally:
            await stream.aclose()

    # ---- ASSETS ----

    async def save(self, *assets: Asset) -> None:
        """Stores an asset on the XR device.
        Args:
            assets: asset objects to be stored on the device.
        Returns:
            None.
        """
        await self._execute_single(CreateOrUpdateAssetsCommand(assets=assets))

    async def list_assets(self) -> list[str]:
        """Lists stored asset keys.
        Returns:
            List of asset identification keys stored on the device.
        """
        return await self._execute_single(ListAssetsCommand())

    async def destroy_asset(self, asset_key: str | None = None, all_assets: bool = False) -> None:
        """Deletes one asset or all assets.
        Args:
            asset_key: Asset key to delete. Ignored if ``all_assets`` is True.
            all_assets: If True, deletes all stored assets.

        Returns:
            None.
        """
        await self._execute_single(DestroyAssetCommand(asset_key=asset_key, all_assets=all_assets))

    async def update(self, *elements: Element) -> None:
        """Creates or updates (upsert) a remote element. Necessary to apply change the state of a remote element instance.
        Args:
            elements: Elements defining the desired state of virtual elements on the client.

        Returns:
            None.
        """
        await self._execute_single(CreateOrUpdateElementCommand(elements=elements))

    async def list_elements(self) -> list[str]:
        """Lists existing elements, both active an inactive.
        Returns:
            List of element identification keys.
        """
        return await self._execute_single(ListElementsCommand())

    async def destroy_element(self, element: Element | None = None, all_elements: bool = False) -> None:
        """Destroys one element or all elements.
        Args:
            element: Element to destroy. If provided, its ``key`` is used.
            all_elements: If True, destroys all elements.

        Returns:
            None.
        """
        await self._execute_single(
            DestroyElementCommand(
                key=element.key if element is not None else None,
                all_elements=all_elements,
            )
        )


class _SyncSenseIter(Iterator[dict[str, Any]]):
    """Blocking iterator over an async generator, running on a given loop."""

    def __init__(self, agen, loop: asyncio.AbstractEventLoop):
        self._agen = agen
        self._loop = loop
        self._done = False

    def __iter__(self) -> "_SyncSenseIter":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._done:
            raise StopIteration
        fut = asyncio.run_coroutine_threadsafe(self._agen.__anext__(), self._loop)
        try:
            return fut.result()
        except StopAsyncIteration:
            self._done = True
            raise StopIteration

    def close(self) -> None:
        if self._done:
            return
        self._done = True
        asyncio.run_coroutine_threadsafe(self._agen.aclose(), self._loop).result()


class SyncXR(AsyncXR):

    def __init__(self, remote: RemoteXRClient, loop: asyncio.AbstractEventLoop, loop_thread: Thread):
        super().__init__(remote)
        self._loop = loop
        self._loop_thread = loop_thread

    def _sync(self, coro):
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("SyncXR called from its event loop thread leads to deadlock")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---- INFO ----
    def info(self) -> DeviceInfo:
        return self._sync(super().info())

    # ---- UI ----
    def write(self, text: str, title: str | None = None) -> None:
        return self._sync(super().write(text=text, title=title))

    def say(self, text: str, title: str | None = None) -> None:
        return self._sync(super().say(text=text, title=title))

    def read(self, text: str, title: str | None = None) -> str:
        return self._sync(super().read(text=text, title=title))

    def passthrough(self, transparency: float) -> None:
        return self._sync(super().passthrough(transparency=transparency))

    # ---- SENSING (SINGLE) ----
    def image(self) -> ImageAsset:
        return self._sync(super().image())

    def virtual_image(self) -> ImageAsset:
        return self._sync(super().virtual_image())

    def depth(self) -> ImageAsset:
        return self._sync(super().depth())

    def eye(self) -> Pose:
        return self._sync(super().eye())

    def head(self) -> Pose:
        return self._sync(super().head())

    def hands(self) -> Hands:
        return self._sync(super().hands())

    # ---- SENSING (STREAM) ----
    def sense(
            self,
            *,
            image: bool = False,
            virtual_image: bool = False,
            depth: bool = False,
            eye: bool = False,
            head: bool = False,
            hands: bool = False,
            rt: bool = True,
    ) -> _SyncSenseIter:
        agen = super().sense(
            image=image,
            virtual_image=virtual_image,
            depth=depth,
            eye=eye,
            head=head,
            hands=hands,
            rt=rt,
        )
        return _SyncSenseIter(agen, self._loop)

    # ---- ENTITIES ----
    def save(self, *assets: Asset) -> None:
        return self._sync(super().save(*assets))

    def list_assets(self) -> list[str]:
        return self._sync(super().list_assets())

    def destroy_asset(self, asset_key: str | None = None, all_assets: bool = False) -> None:
        return self._sync(super().destroy_asset(asset_key=asset_key, all_assets=all_assets))

    def update(self, *element: Element) -> None:
        return self._sync(super().update(*element))

    def list_elements(self) -> list[str]:
        return self._sync(super().list_elements())

    def destroy_element(self, element: Element | None = None, all_elements: bool = False) -> None:
        return self._sync(super().destroy_element(element=element, all_elements=all_elements))


for _name, _member in list(SyncXR.__dict__.items()):
    if _name.startswith("_"):
        continue
    _src = getattr(AsyncXR, _name, None)
    if _src is not None and getattr(_src, "__doc__", None):
        getattr(SyncXR, _name).__doc__ = _src.__doc__
