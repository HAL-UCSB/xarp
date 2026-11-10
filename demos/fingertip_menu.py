from collections.abc import Callable, Sequence
from dataclasses import dataclass

from xarp.data_models import Hands
from xarp.entities import Element, TextAsset
from xarp.express import AsyncGeneratorIterator, SyncXR
from xarp.gestures import (
    PALM,
    DIGIT_TIPS,
    pinch_any
)
from xarp.spatial import Pose, Transform, Vector3
from xarp.time import utc_ts
from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run

DEFAULT_HAND = "left"
PALM_VISIBILITY_THRESHOLD = 0.5

HOLD_DURATION_MS = 3000
PROGRESS_UPDATE_MS = 1000
PROGRESS_STEP_COUNT = HOLD_DURATION_MS // PROGRESS_UPDATE_MS
FILLED_PROGRESS_CIRCLE = "\u25cf"
EMPTY_PROGRESS_CIRCLE = "\u25cb"


@dataclass
class HoldState:
    """Tracks which fingertip is being held, and when the hold began."""

    tip: str | None = None
    started_at_ms: int | None = None

    @property
    def is_active(self) -> bool:
        return self.tip is not None and self.started_at_ms is not None

    def start(self, tip: str, now_ms: int) -> None:
        self.tip = tip
        self.started_at_ms = now_ms

    def reset(self) -> None:
        self.tip = None
        self.started_at_ms = None

    def elapsed_ms(self, now_ms: int) -> int:
        if self.started_at_ms is None:
            return 0
        return max(0, now_ms - self.started_at_ms)


def is_palm_visible_to_eye(eye: Pose, hand: tuple[Pose, ...]) -> bool:
    gaze_forward = eye.rotation.rotate_vector(Vector3.forward())
    palm_normal = hand[PALM].rotation.rotate_vector(Vector3.up())
    return Vector3.dot(gaze_forward, palm_normal) > PALM_VISIBILITY_THRESHOLD


def progress_filled_count(elapsed_ms: int) -> int:
    elapsed_steps = elapsed_ms // PROGRESS_UPDATE_MS
    return min(PROGRESS_STEP_COUNT, max(0, elapsed_steps))


def progress_circles(elapsed_ms: int) -> str:
    filled_count = progress_filled_count(elapsed_ms)
    empty_count = PROGRESS_STEP_COUNT - filled_count
    return f"{FILLED_PROGRESS_CIRCLE * filled_count}{EMPTY_PROGRESS_CIRCLE * empty_count}"


def create_option_elements(options: Sequence[str]) -> dict[str, Element]:
    return {
        option: Element(
            key=option,
            asset=TextAsset.from_obj(option, asset_key=option),
        )
        for option in options
    }


def hide_option_elements(xr: SyncXR, option_elements: dict[str, Element]) -> None:
    for option_element in option_elements.values():
        xr.destroy_element(option_element)


def pinched_tip(hand: tuple[Pose, ...], tips: Sequence[str] = DIGIT_TIPS) -> str | None:
    return next((tip for tip in tips if pinch_any(hand, tip)), None)


def update_hold_state(
        hold: HoldState,
        current_tip: str | None,
        now_ms: int,
) -> None:
    """Keep a hold active only while the same fingertip remains pinched."""

    if hold.tip is not None and hold.tip != current_tip:
        hold.reset()

    if hold.tip is None and current_tip is not None:
        hold.start(current_tip, now_ms)


def option_for_tip(tip: str | None, options: Sequence[str], tips: Sequence[str]) -> str | None:
    if tip is None or tip not in tips:
        return None

    option_index = tips.index(tip)
    if option_index >= len(options):
        return None

    return options[option_index]


def option_label(option: str, tip: str, hold: HoldState, held_ms: int) -> str:
    if hold.tip != tip:
        return option
    return f"{option} {progress_circles(held_ms)}"


def option_asset_key(option: str, tip: str, hold: HoldState, held_ms: int) -> str:
    if hold.tip != tip:
        return option

    # The XR client can cache text assets, so each visual step gets a stable key.
    return f"{option}:progress:{progress_filled_count(held_ms)}"


def update_option_element(
        xr: SyncXR,
        option_element: Element,
        option: str,
        tip: str,
        hand: tuple[Pose, ...],
        eye: Pose,
        hold: HoldState,
        held_ms: int,
) -> None:
    option_element.asset = TextAsset.from_obj(
        option_label(option, tip, hold, held_ms),
        asset_key=option_asset_key(option, tip, hold, held_ms),
    )
    option_element.transform = Transform(
        position=hand[tip].position,
        rotation=eye.rotation,
    )
    xr.update(option_element)


def display_fingertip_menu(
        xr: SyncXR,
        stream: AsyncGeneratorIterator,
        options: Sequence[str],
        on_select: Callable[[str], None],
        hand_key=DEFAULT_HAND,
) -> None:
    hold = HoldState()
    option_elements = create_option_elements(options)
    menu_tips = DIGIT_TIPS[:len(options)]

    for frame in stream:
        hands: Hands = frame["hands"]
        eye: Pose = frame["eye"]
        hand = hands[hand_key]

        if not hand:
            hide_option_elements(xr, option_elements)
            return

        if not is_palm_visible_to_eye(eye, hand):
            hide_option_elements(xr, option_elements)
            return

        now_ms = utc_ts()
        update_hold_state(hold, pinched_tip(hand, menu_tips), now_ms)

        held_ms = min(hold.elapsed_ms(now_ms), HOLD_DURATION_MS)
        selected_option = (
            option_for_tip(hold.tip, options, menu_tips)
            if hold.is_active and held_ms >= HOLD_DURATION_MS
            else None
        )

        for tip, option in zip(menu_tips, options):
            update_option_element(
                xr=xr,
                option_element=option_elements[option],
                option=option,
                tip=tip,
                hand=hand,
                eye=eye,
                hold=hold,
                held_ms=held_ms,
            )

        if selected_option is not None:
            on_select(selected_option)
            hold.reset()


APP_OPTIONS = ["option A", "option B", "option C", "option D"]


def app(xr: SyncXR, *args, **kwargs):
    stream = xr.sense(eye=True, hands=True)

    def handle_selection(_option: str) -> None:
        xr.say(_option)

    while True:
        display_fingertip_menu(
            xr=xr,
            stream=stream,
            options=APP_OPTIONS,
            on_select=handle_selection,
        )


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
