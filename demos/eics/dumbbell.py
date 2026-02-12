from smolagents import OpenAIServerModel, MultiStepAgent

from xarp.agents import run_xr_agent
from xarp.entities import GLBAsset, Element
from xarp.express import SyncXR
from xarp.gestures import victory, PALM
from xarp.server import show_qrcode_link
from xarp.spatial import Transform, Vector3, Pose

model = OpenAIServerModel(
    api_base="http://128.111.28.83:1234/v1",
    model_id="mistralai/devstral-small-2-2512",
    api_key="sk-lm-a6zNe4ph:rgEl9stEQsIy55wxw9L4"
)


def app(xarp: SyncXR, agent: MultiStepAgent, params):
    glb_asset = GLBAsset(asset_key="asset_dumbbell")
    with open(r"D:\Arthur Data\xarp\demos\eics\dumbbell.glb", "rb") as f:
        glb_asset.raw = f.read()
    xarp.save(glb_asset)
    glb_asset.raw = None

    user_dumbbell = Element(
        key="user_dumbbell",
        asset=glb_asset
    )

    while True:
        stream = xarp.sense(hands=True)
        rotations = []
        positions = []
        recording = False
        for frame in stream:
            hands = frame["hands"]
            if hands.left and victory(hands.left):
                if recording:
                    break
                recording = True
                xarp.destroy_element(all_elements=True)
                xarp.say("Tracking")
                user_dumbbell.active = True
            if recording and hands.right:
                palm: Pose = hands.right[PALM]
                positions.append(palm.position.to_numpy().tolist())
                rotations.append(palm.rotation.to_euler_angles().to_numpy().tolist())
                user_dumbbell.transform = Transform(
                    position=palm.position,
                    rotation=palm.rotation,
                    scale=Vector3.one() * .3)
                xarp.update(user_dumbbell)

        stream.close()
        user_dumbbell.active = False
        xarp.update(user_dumbbell)
        xarp.say("Analyzing")

        agent.run(
            f'Analyze the palm positions and rotations during a standard dumbbell curl and find the most critical mistake. Create a visual demonstration using a combination of labels, primitives, and colors to help the user improve their technique. Make a concise suggestion using speech.',
            additional_args=dict(
                positions=positions[1::4],
                rotations=rotations[1::4],
            )
        )


if __name__ == '__main__':
    show_qrcode_link()
    run_xr_agent(app, model, add_base_tools=True)
