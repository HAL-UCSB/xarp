from smolagents import MultiStepAgent, OpenAIServerModel

from xarp.agents import run_xr_agent
from xarp.express import SyncXR
from xarp.server import show_qrcode_link

model = OpenAIServerModel(
    model_id="gpt-5-mini",
    api_key="lm-studio",
    api_base="http://IP:PORT/v1"
)

custom_system_prompt = """
You are an agent with extended reality tools. You can sense the environment and display information.
Before asking me about for extra information, use your tools to understand the context.
The user cannot read the output of "print" functions, use the "write" or "say" tools instead.
"""

def xr_agent_app(xr: SyncXR, agent: MultiStepAgent, params):
    agent.prompt_templates["system_prompt"] = custom_system_prompt + agent.prompt_templates["system_prompt"]
    xr.image().obj.show()
    while True:
        xr.say("How can I help you?")
        request = xr.read()
        answer = agent.run(request)
        xr.write(answer)


if __name__ == '__main__':
    show_qrcode_link()
    run_xr_agent(xr_agent_app, model)
