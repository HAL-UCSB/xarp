from datetime import time

from openai import OpenAI

client = OpenAI(
    api_key='')

with open(
        r"C:\Users\Arthur\Downloads\water-flows-out-of-the-pipes-into-the-green-rice-fields-free-photo-4173837242.jpg",
        "rb") as f:
    # Generate the video
    video = client.videos.create(
        prompt="water flowing",
        input_reference=f
    )


def wait_for_video_to_finish(video_id, poll_interval=5, timeout=600):
    """
    Poll status until the video is ready or timeout is reached.
    Returns the final job info.
    """
    elapsed = 0  # Keep track of the elapsed time
    while elapsed < timeout:
        job = client.videos.retrieve(video_id)
        status = job.status
        progress = job.progress
        print(f"Status: {status}, {progress}%")
        if status == "completed":
            return job
        if status == "failed":
            raise RuntimeError(f"Failed to generate the video: {job.error.message}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise RuntimeError("Polling timed out")


def download_video(video_id):
    response = client.videos.download_content(
        video_id=video_id,
    )
    video_bytes = response.read()
    with open(f"{video_id}.mp4", "wb") as f:
        f.write(video_bytes)


wait_for_video_to_finish(video.id)
download_video(video.id)
