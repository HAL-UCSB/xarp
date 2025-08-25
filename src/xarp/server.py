import asyncio
import threading

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from smolagents import ActionStep, AgentError
from smolagents import OpenAIServerModel, CodeAgent
from starlette.exceptions import WebSocketException
from starlette.websockets import WebSocketDisconnect

from xarp import XRApp, settings
from xarp.authorization import is_authorized
from xarp.models import Session
from xarp.storage.local_file_system import SessionFileRepository

model = OpenAIServerModel(
    model_id=settings.model_id,
    api_base=settings.model_api_base,
    api_key=settings.model_api_key)

app = FastAPI()


@app.get('/')
async def root():
    return 'ok'


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket, user_id: str, session_ts: int = None):
    if not is_authorized(user_id):
        raise HTTPException(
            status_code=401,
            detail='Unauthorized user')

    sessions = SessionFileRepository(settings.local_storage / user_id)
    if session_ts:
        result_sessions = sessions.load(session_ts)
        session = result_sessions[0] if result_sessions else Session(ts=session_ts)
    else:
        session = Session()

    await ws.accept()
    await ws.send_text(str(session.ts))

    loop = asyncio.get_running_loop()
    xr = XRApp(ws, loop, session)
    completion_event = asyncio.Event()
    tools = xr.as_tools()
    agent = CodeAgent(tools=tools, model=model, add_base_tools=False)

    def agent_entrypoint():
        while not completion_event.is_set():
            try:
                meters = xr.depth()
                img = xr.see()
                img.show()
                xr.write('How can I help you?')
                request = xr.read()
                # xr.see()
                answer = agent.run(
                    request,
                    reset=True,
                    images=xr._imgs
                )
                xr.write(answer)
            except (WebSocketException, RuntimeError) as e:
                print('WebSocket Error:', e)
                completion_event.set()
            except AgentError as e:
                print('Agent Error: ', e)

    def set_observations_images(step: ActionStep):
        if xr.called_see():
            step.observations_images = [img.copy() for img in xr._imgs]
            xr._imgs.clear()

    def stop_when_disconnected(step: ActionStep):
        if step.error and 'websocket' in step.error.message.lower():
            raise WebSocketDisconnect(step.error.message)

    agent.step_callbacks = [
        lambda *args, **kwargs: sessions.save(session),
        stop_when_disconnected,
        set_observations_images
    ]

    thread = threading.Thread(target=agent_entrypoint, daemon=True)
    thread.start()
    await completion_event.wait()
    sessions.save(session)

    try:
        await ws.close(code=1000)
    except (WebSocketException, RuntimeError):
        pass


if __name__ == '__main__':
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port)
