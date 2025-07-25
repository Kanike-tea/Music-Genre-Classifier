import subprocess

def convert_to_wav(src_path, dst_path):
    subprocess.call(['ffmpeg', '-y', '-i', src_path, dst_path])