import subprocess
import json

def test_ffprobe():
    try:
        args = ['ffprobe', '-version']
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode == 0:
            print("ffprobe found and working.")
            print(out.decode('utf-8').split('\n')[0])
        else:
            print(f"ffprobe returned error: {err.decode('utf-8')}")
    except FileNotFoundError:
        print("ffprobe NOT found in PATH.")
    except Exception as e:
        print(f"ffprobe test failed: {e}")

test_ffprobe()
