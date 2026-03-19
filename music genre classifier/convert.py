import subprocess
import shutil


def convert_to_wav(src_path: str, dst_path: str) -> None:
    """Convert any audio file to WAV using ffmpeg.

    Raises:
        EnvironmentError: if ffmpeg is not installed or not on PATH.
        RuntimeError: if the conversion fails.
    """
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not on PATH. "
            "Install it from https://ffmpeg.org/download.html"
        )

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, dst_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for '{src_path}':\n"
            + result.stderr.decode(errors="replace")
        )
