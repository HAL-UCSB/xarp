import base64
import json
import threading
from functools import partial
from queue import Queue

import streamlit as st
from PIL import Image
from streamlit.runtime import Runtime
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx
from websockets import WebSocketException
from websockets.sync.client import connect

from xarp.data_models import ChatMessage
from xarp.data_models.spatial import Transform
from xarp.storage.local_file_system import UserRepositoryLocalFileSystem
from xarp.web.chat_ui import render_chat_messages, require_session_obj
from xarp import XRCommand, settings

ws_key = 'ws'
chat_key = 'chat'
executed_key = 'executed'
inbound_key = 'inbound'
outbound_key = 'outbound'
threads_key = 'threads'
main_session_id_key = 'main_session_id'


def disconnect():
    if ws_key in st.session_state:
        st.session_state[ws_key].close()

    session_keys_to_del = ws_key, executed_key, inbound_key, outbound_key, threads_key
    for k in session_keys_to_del:
        if k in st.session_state:
            del st.session_state[k]


ctx = get_script_run_ctx()
st.session_state[main_session_id_key] = ctx.session_id


def rerun_streamlit_thread():
    """https://github.com/streamlit/streamlit/issues/2838#issuecomment-1603143615"""
    main_session_id = st.session_state[main_session_id_key]
    runtime = Runtime.instance()
    sessions = runtime._session_mgr.list_sessions()
    for s in sessions:
        if s.session.id == main_session_id:
            s.session._handle_rerun_script_request()
            return


url = st.text_input('Websocket Server URL', value=r'ws://127.0.0.1:8000/ws')

users = UserRepositoryLocalFileSystem(settings.local_storage)
all_users = list(users.all())
user = st.selectbox('User ID', all_users, format_func=lambda u: u.user_id)
session = st.selectbox('Session', [None] + user.sessions,
                       format_func=lambda s: str(s.ts) if s else 'New Session')

top = st.container()
bottom = st.container()

if top.toggle('connect'):

    url += f'?user_id={user.user_id}'
    if session:
        url += f'&ts={session.ts}'

    st.warning(url)

    if session:
        chat = require_session_obj(chat_key, session.chat)
    else:
        chat = require_session_obj(chat_key, [])

    inbound = require_session_obj(inbound_key, list)
    outbound = require_session_obj(outbound_key, Queue)
    threads = require_session_obj(threads_key, list)

    if not threads:

        ws = require_session_obj(ws_key, partial(connect, url))
        ts = ws.recv()


        def run_inbound():
            while ws_key in st.session_state:
                try:
                    cmd = ws.recv()
                    cmd_json = json.loads(cmd)
                    _xr_cmd = XRCommand.model_validate_json(cmd_json)
                    inbound.append(_xr_cmd)
                    rerun_streamlit_thread()
                except WebSocketException as e:
                    st.error(e)
                    disconnect()


        def run_outbound():
            while ws_key in st.session_state:
                try:
                    if not outbound.empty():
                        out_msg = outbound.get() or '_'
                        ws.send(out_msg)
                        rerun_streamlit_thread()
                except WebSocketException as e:
                    st.error(e)
                    disconnect()


        threads.append(threading.Thread(target=run_inbound, daemon=True))
        threads.append(threading.Thread(target=run_outbound, daemon=True))
        for thread in threads:
            add_script_run_ctx(thread, ctx)
            thread.start()

    executed = require_session_obj(executed_key, set)
    for i, xr_cmd in enumerate(inbound):
        if i in executed:
            continue
        match xr_cmd.cmd:
            case 'write':
                text = '\n'.join(map(str, xr_cmd.args))
                msg = ChatMessage.from_assistant(text.strip())
                chat.append(msg)
                outbound.put_nowait('')
                executed.add(i)
            case 'read':
                if text := bottom.chat_input(key=f'chat_input_{xr_cmd.ts}'):
                    msg = ChatMessage.from_user(text)
                    chat.append(msg)
                    outbound.put_nowait(text)
                    bottom.empty()
                    executed.add(i)
            case 'head':
                with bottom:
                    pose = Transform.from_euler_xyz((
                        st.slider('x angle', min_value=0, max_value=360),
                        st.slider('y angle', min_value=0, max_value=360),
                        st.slider('z angle', min_value=0, max_value=360)
                    ))
                    pose.position = (
                        st.number_input('x position'),
                        st.number_input('y position'),
                        st.number_input('z position')
                    )

                    if st.button('send'):
                        pose_json = pose.model_dump_json()
                        msg = ChatMessage.from_user(pose_json)
                        chat.append(msg)
                        outbound.put_nowait(pose_json)
                        bottom.empty()
                        executed.add(i)

            case 'image':
                if img_files := bottom.file_uploader('see', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True):
                    for i, src in enumerate(img_files):
                        path = settings.local_storage / str(user) / str(session) / src.name
                        with open(path) as dest:
                            dest.write(src.read())

                    msg = ChatMessage.from_user('', files=img_files)
                    chat.append(msg)

                    response = []
                    for img_file in img_files:
                        img = Image.open(img_file).convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                        width, height = img.size
                        pixel_bytes = img.tobytes()
                        b64_pixels = base64.b64encode(pixel_bytes).decode('utf-8')

                        img_dict = {
                            "pixels": b64_pixels,
                            "width": width,
                            "height": height
                        }
                        response.append(img_dict)

                    response_json = json.dumps(response)
                    outbound.put_nowait(response_json)
                    bottom.empty()
                    executed.add(i)
            case _:
                text = '[Unsupported Command]\n' + '\n'.join(map(str, xr_cmd.args))
                st.error(text)
                outbound.put_nowait('')
                executed.add(i)



else:
    chat = session.chat if session else []
    if ws_key in st.session_state:
        disconnect()
with top:
    with st.container(border=True, height=400):
        render_chat_messages(chat)
