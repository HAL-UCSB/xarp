import base64
import io

import PIL
from smolagents import OpenAIServerModel, MultiStepAgent

from xarp.agents import run_xr_agent
from xarp.express import SyncXR
from xarp.gestures import victory
from xarp.server import show_qrcode_link

model = OpenAIServerModel(
    api_base="http://128.111.28.74:1234/v1",
    model_id="mistralai/devstral-small-2-2512",
    api_key="sk-lm-lnrIBXgS:27lmZnE7f95x4h5BQSgD"
)


def vlm(request: str, img: PIL.Image.Image) -> str:
    # Encode PIL image as PNG -> base64 data URL
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    # OpenAIServerModel is callable: model(messages, stop_sequences=...)
    out = model(messages)

    return out.content if hasattr(out, "content") else str(out)


prompt = """"Capture an image and identify in which stage of a brewing coffee procedure the user is. Display text instructions for the next step in a label. The label must be as close as possible to where the next step must be executed.""""


def app(xarp: SyncXR, agent: MultiStepAgent, params):
    for frame in xarp.sense(hands=True):
        hands = frame["hands"]
        # Pinch gesture test
        if hands.right and victory(hands.right):
            answer = agent.run(prompt)
            xarp.say(answer)


if __name__ == '__main__':
    show_qrcode_link()
    run_xr_agent(app, model)
