import csv, json, time
from pathlib import Path

from xarp.express import SyncXR
from xarp.server import run, make_qrcode_image


def pose(p):
    return [p.position.x, p.position.y, p.position.z,
            p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w] if p else []


def app(xr: SyncXR, *args, **kwargs):
    out = Path("session_" + time.strftime("%Y%m%d_%H%M%S"))
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    f = open(out / "frames.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["i", "t", "head", "left", "right", "image", "depth", "depth_bytes"])
    print("recording ->", out.resolve())

    stream = xr.sense(depth=True, image=True, head=True, hands=True, rt=False)
    t0 = time.time()
    i = 0
    try:
        for frame in stream:
            img = out / "image" / f"{i:06d}.jpg"
            img.write_bytes(frame["image"].raw)

            # depth is a raw uint16 buffer, not an encoded image, so dump the bytes
            raw = frame["depth"].raw
            dep = out / "depth" / f"{i:06d}.bin"
            dep.write_bytes(raw)

            hands = frame["hands"]
            w.writerow([
                i, round(time.time() - t0, 4),
                json.dumps(pose(frame["head"])),
                json.dumps([pose(p) for p in hands.left]),
                json.dumps([pose(p) for p in hands.right]),
                img.relative_to(out).as_posix(),
                dep.relative_to(out).as_posix(),
                len(raw),
            ])
            f.flush()
            i += 1
    finally:
        stream.close()
        f.close()
        print(i, "frames ->", out.resolve())


if __name__ == '__main__':
    make_qrcode_image()
    run(app)