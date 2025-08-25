import asyncio
import base64
import io
import json
import os
import pathlib
from asyncio import AbstractEventLoop
from typing import ClassVar

import PIL
import numpy as np
from PIL import Image
from fastapi import WebSocket
from mcp.server.fastmcp import FastMCP
from pydantic_settings import BaseSettings, SettingsConfigDict
from smolagents import tool, AgentImage

from xarp.models import XRCommand, Session, ChatMessage, ChatFile, Pose


class Settings(BaseSettings):
    # model
    model_api_base: str
    model_id: str
    model_api_key: str

    # authorization
    authorization_file_path: pathlib.PurePosixPath

    # server
    server_port: int
    server_host: str

    # storage
    local_storage: pathlib.PurePosixPath

    _ENV_VAR_XRLIT_DOT_ENV: ClassVar[str] = 'XRLIT_DOT_ENV'
    _DEFAULT_XRLIT_DOT_ENV: ClassVar[pathlib.Path] = pathlib.Path('.env')

    model_config = SettingsConfigDict(
        env_file=os.environ.get(
            _ENV_VAR_XRLIT_DOT_ENV,
            _DEFAULT_XRLIT_DOT_ENV),
        env_file_encoding='utf-8',
        extra='ignore')


settings = Settings()


class AsyncXRApp:

    def __init__(self, ws: WebSocket, session: Session):
        self._ws = ws
        self._called_see = False
        self._imgs = list()
        self._session = session

    def called_see(self):
        if self._called_see:
            self._called_see = False
            return True
        return False

    async def write(self, text: str) -> None:
        """
        Displays a readable text message to the user during intermediary steps of execution and planning.
        Do not use to communicate final answers.
        Args:
            text (str): text message to be displayed
        """
        cmd = XRCommand(
            cmd='write',
            args=[text]
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)
        await self._ws.receive()

        chat_message = ChatMessage.from_assistant(text)
        self._session.chat.messages.append(chat_message)
        chat_message_json = chat_message.model_dump_json()
        print(f'I told the user: "{text}"')

    async def read(self) -> str:
        """
        Reads a text input from the user. Use this tool instead of input() when you need to read a text input from the user.
        With this tool you can read the answer from a question you previously asked.
        Returns:
            str: text input from the user.
        """
        cmd = XRCommand(
            cmd='read'
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)
        text = await self._ws.receive_text()
        await self._ws.receive()

        chat_message = ChatMessage.from_user(text)
        self._session.chat.messages.append(chat_message)
        chat_message_json = chat_message.model_dump_json()
        print(f'The user told me: "{text}"')

        return text

    async def depth(self) -> np.array:
        """
        A matrix with depth in meters measured from the user's point of view. Useful when you need to estimate depth.
        Returns:
            np.array: depth matrix
        """
        cmd = XRCommand(
            cmd='depth'
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)

        img_dicts = await self._ws.receive_json()
        for img_dict in img_dicts:
            size = img_dict['width'], img_dict['height']
            meters = np.array(img_dict['meters']).reshape(size)
            pixels = 255 * (meters / np.max(meters))
            img = Image.fromarray(pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img.show()

        await self._ws.receive()

        chat_img_files = []
        for img in self._imgs:
            img_format = img.format or 'png'
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img_format)
            chat_img_file = ChatFile(
                mime_type=f'image/{img_format}',
                raw=img_bytes)
            chat_img_files.append(chat_img_file)
        chat_message = ChatMessage.from_user('', files=chat_img_files)
        self._session.chat.messages.append(chat_message)

        return meters


    async def see(self) -> PIL.Image.Image:
        """
        Observes the environment from the user's point of view. Useful when you need a visual of the surroundings.
        Returns:
            PIL.Image.Image: image
        """
        cmd = XRCommand(
            cmd='see'
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)

        img_dicts = await self._ws.receive_json()
        for img_dict in img_dicts:
            pixels = base64.b64decode(img_dict['pixels'])
            size = img_dict['width'], img_dict['height']
            img = Image.frombytes('RGBA', size, pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            # img_base64 = base64.b64encode(img).decode('utf-8')
            # img.show()
            self._imgs.append(img)

        await self._ws.receive()
        self._called_see = True

        chat_img_files = []
        for img in self._imgs:
            img_format = img.format or 'png'
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img_format)
            chat_img_file = ChatFile(
                mime_type=f'image/{img_format}',
                raw=img_bytes)
            chat_img_files.append(chat_img_file)
        chat_message = ChatMessage.from_user('', files=chat_img_files)
        self._session.chat.messages.append(chat_message)

        print(f'I have {"1 image" if len(self._imgs) == 1 else str(len(self._imgs)) + " images"} to observe.')
        return self._imgs[-1]

    async def head_pose(self) -> Pose:
        """
        Gets the current user's head pose (XYZ position and quaternion rotation). Useful to understand where the user is in space or
        where they are looking at.
        Returns:
            xrlit.models.Pose: user's head pose (XYZ position and quaternion rotation)
        """
        cmd = XRCommand(
            cmd='head_pose'
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)
        head_json = await self._ws.receive_text()
        print(f"this is the head_json: {head_json}")
        await self._ws.receive()

        pose = Pose.model_validate_json(head_json)

        chat_message = ChatMessage.from_user(head_json)
        self._session.chat.messages.append(chat_message)
        chat_message_json = chat_message.model_dump_json()
        print(chat_message_json)
        return pose

    # # the docstrings become the tool descriptions once read by smolagent

    async def mesh(self) -> str:
        """
        Retrieves the spatial mesh of the user's environment. This mesh provides a 3D representation of the user's physical surroundings, including walls, floors, furniture, and obstacles.
        It is useful for spatial reasoning tasks, such as navigation, object placement, and scene understanding.
        Use this tool when you need to analyze the layout of a room, detect walkable surfaces, or align virtual content with real-world geometry.
        Returns:
            str: Retrieve back a JSON containing the geometry of the requested component in the user's physical space, with dimensions in meters.
        """
        # """
        # Retrieves the spatial mesh of the user's environment. This mesh provides a 3D representation of the user's physical surroundings, including walls, floors, furniture, and obstacles.
        # It is useful for spatial reasoning tasks, such as navigation, object placement, and scene understanding.
        # Use this tool when you need to analyze the layout of a room, detect walkable surfaces, or align virtual content with real-world geometry.
        # Returns:
        #     str: Retrieve back a JSON containing the layout of the room, where the furnitures, exits, doors, walls, and other components are located. 
        # """

        cmd = XRCommand(
            cmd='mesh'
        ).model_dump_json()
        sys_message = ChatMessage.from_system(cmd)
        self._session.chat.messages.append(sys_message)
        await self._ws.send_json(cmd)

        print("[SERVER] Sent mesh command, waiting for client response...")

        # First receive the header
        header = await self._ws.receive_json()
        if header.get('type') != 'mesh_header':
            raise ValueError("Unexpected response format")

        expected_size = header['size']
        print(f"Expecting mesh data of size: {expected_size}")

        # Then receive the actual data
        text = await self._ws.receive_text()
        print(f"Received {len(text)} bytes of mesh data")

        # Verify size matches
        if len(text) != expected_size:
            raise ValueError(f"Data size mismatch. Expected {expected_size}, got {len(text)}")

        await self._ws.receive()  # Final acknowledgement

        mesh_data = json.loads(text)

        floor_uuid = mesh_data["Rooms"][0]["RoomLayout"]["FloorUuid"]
        # maybe we can get the type of room components based on user request. i.e. getting Floor, Walls, Table depending on user req
        anchors = mesh_data["Rooms"][0]["Anchors"]

        floor_anchor = next((a for a in anchors if a["UUID"] == floor_uuid), None)
        print(f"this is the floor anchor: {floor_anchor}")

        chat_message = ChatMessage.from_user(json.dumps(floor_anchor))
        # chat_message = ChatMessage.from_user(json.dumps(mesh_data))
        self._session.chat.messages.append(chat_message)
        chat_message_json = chat_message.model_dump_json()
        print(chat_message_json)

        return str(floor_anchor)
        # return str(mesh_data)

    def as_mcp_server(self):
        mcp = FastMCP('XRLitApp')
        mcp.add_tool(self.read)
        mcp.add_tool(self.write)
        mcp.add_tool(self.head_pose)
        mcp.add_tool(self.see)
        mcp.add_tool(self.mesh)
        return mcp


class XRApp(AsyncXRApp):

    def __init__(self, ws: WebSocket, loop: AbstractEventLoop, session: Session):
        super().__init__(ws, session)
        self._loop = loop

    def _sync(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def _raise(self) -> None:
        """
        raises exception.
        """

        async def __raise():
            print('the exception was raised.')
            raise Exception('ops')

        coroutine = __raise()
        return self._sync(coroutine)

    def read(self) -> str:
        coroutine = super().read()
        return self._sync(coroutine)

    def write(self, text: str) -> None:
        coroutine = super().write(text)
        return self._sync(coroutine)

    def head_pose(self) -> Pose:
        coroutine = super().head_pose()
        return self._sync(coroutine)

    def see(self) -> PIL.Image.Image:
        coroutine = super().see()
        img = self._sync(coroutine)
        # img.show()
        return img

    def depth(self) -> np.array:
        coroutine = super().depth()
        meters = self._sync(coroutine)
        return meters

    def mesh(self) -> str:
        coroutine = super().mesh()
        return self._sync(coroutine)

    def as_tools(self):
        methods = [
            self.read,
            self.write,
            self.head_pose,
            self.mesh
        ]
        tools = [tool(t) for t in methods]

        see = tool(self.see)
        see.output_type = AgentImage.__name__
        see.forward = lambda *args, **kwargs: AgentImage(self.see(*args, **kwargs))
        tools.append(see)

        return tools
