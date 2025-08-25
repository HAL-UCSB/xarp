import json
import pathlib
import re
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from xarp.models import ChatMessage, ChatFile, Chat


def require_session_obj(key, default):
    if key in st.session_state:
        return st.session_state[key]
    value = default() if callable(default) else default
    st.session_state[key] = value
    return value


def render_file(file: ChatFile):
    if re.match('image/.*', file.mime_type):
        st.image(file.raw or str(file.path), use_container_width=True)
        return

    if re.match('text/csv', file.mime_type):
        file.raw.seek(0)
        df = pd.read_csv(file.raw).head()
        st.dataframe(df)
        return

    prefix = file.mime_type.split('/')[0]
    if hasattr(st, prefix):
        render_fn = getattr(st, prefix)
        render_fn(file.raw)


def render_chat_message(i: int, message: ChatMessage):
    with st.chat_message(message.role):
        if message.content.text:
            try:
                json_data = json.loads(message.content.text)
                st.json(json_data)
            except JSONDecodeError:
                st.markdown(message.content.text)

        for file in message.content.files:
            render_file(file)

            st.download_button(
                label=(file.raw or file.path).name,
                key=f'download_button_{id(file)}',
                icon=':material/download:',
                file_name=(file.raw or file.path).name,
                data=file.raw or pathlib.Path(*file.path.parts).read_bytes(),
                mime=file.mime_type)


def save_files(files, persistent_path: Path) -> list[Path]:
    paths = []
    for file in files:
        path = persistent_path / file.name
        path.write_bytes(file.read())
        paths.append(path)
    paths


def uploaded_files_to_chat_files(files: list[UploadedFile]) -> list[ChatFile]:
    return [ChatFile(mime_type=f.type, raw=f, path=f.name) for f in files]


@st.fragment
def render_chat_messages(chat: Chat):
    for i, message in enumerate(chat.messages):
        render_chat_message(i, message)
