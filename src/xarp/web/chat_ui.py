import pathlib
from typing import List

import pandas as pd
import streamlit as st

from xarp.resources import ImageResource
from xarp.chat import ChatMessage


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
            st.image(f.read(), width='stretch')
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
        for content in message.content:

            if isinstance(content, ImageResource):
                with open(content.path.as_posix(), 'rb') as f:
                    st.image(f.read(), width='stretch')
                    continue

            content_json = content.model_dump_json()
            st.json(content_json, expanded=False)


@st.fragment
def render_chat_messages(chat: List[ChatMessage]):
    for i, message in enumerate(chat):
        render_chat_message(i, message)
