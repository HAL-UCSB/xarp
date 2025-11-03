import json
import pathlib
from json import JSONDecodeError
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from xarp.data_models import ChatMessage


def require_session_obj(key, default):
    if key in st.session_state:
        return st.session_state[key]
    value = default() if callable(default) else default
    st.session_state[key] = value
    return value


def render_file_preview(path: pathlib.PurePath):
    # display preview
    extension = path.suffix[1:]
    if extension in ('png', 'jpg', 'jpeg'):
        with open(path.as_posix(), 'rb') as f:
            st.image(f.read(), use_container_width=True)
    elif extension == 'json':
        with open(path) as f:
            st.json(f.read())
    elif path.suffix == 'csv':
        with open(path) as f:
            f.seek(0)
            df = pd.read_csv(f).head()
            st.dataframe(df)
            return


def render_chat_message(i: int, message: ChatMessage):
    with st.chat_message(message.role):
        for _text in message.content.text:
            try:
                json_data = json.loads(_text)
                st.json(json_data)
            except JSONDecodeError:
                st.markdown(_text)

        for path in message.content.files:
            if str(path).startswith('file:'):
                path = pathlib.PurePath(str(path)[6:])

            render_file_preview(path)

            # download
            with open(path, 'rb') as f:
                raw = f.read()

            st.download_button(
                label='Download',
                data=raw,
                file_name=path.name,
                icon=':material/download:',
                key=f'download_button_{message.ts}',
            )


def save_files(files, persistent_path: Path) -> List[pathlib.Path]:
    paths = []
    for file in files:
        path = persistent_path / file.name
        path.write_bytes(file.read())
        paths.append(path)
    return paths


@st.fragment
def render_chat_messages(chat: List[ChatMessage]):
    for i, message in enumerate(chat):
        render_chat_message(i, message)
