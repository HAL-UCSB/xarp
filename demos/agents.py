import os
import textwrap
from typing import Any

from smolagents import MultiStepAgent, OpenAIModel

from xarp.agents import run_xr_agent
from xarp.express import SyncXR
from xarp.server import make_qrcode_image


DEFAULT_TASK = "Continuously place cubes at the user’s head position, with each cube’s color determined by the user’s movement speed."


def make_model() -> OpenAIModel:
    """Create the demo model from environment variables."""
    return OpenAIModel(
        model_id=os.getenv("XARP_AGENT_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
        organization=os.getenv("OPENAI_ORG_ID"),
        project=os.getenv("OPENAI_PROJECT_ID"),
    )


def demo(xr: SyncXR, agent: MultiStepAgent, params: dict[str, Any]) -> None:
    task = params.get("task") or os.getenv("XARP_AGENT_TASK", DEFAULT_TASK)

    xr.destroy_element(all_elements=True)
    xr.passthrough(1.0)
    xr.write("Starting XR agent demo...", title="XARP Agent")
    result = agent.run(task)
    xr.write(str(result), title="Agent finished", hide_after_seconds=10)


if __name__ == "__main__":
    make_qrcode_image()
    run_xr_agent(demo, make_model())
