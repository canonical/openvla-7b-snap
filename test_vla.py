import argparse
import time

import json_numpy
import numpy as np
import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9090/act")
    parser.add_argument("--image", default="lego_on_table.png")
    parser.add_argument("--prompt", default="pick up blue block")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    proprio = np.zeros(7, dtype=np.float32)

    payload = {
        "proprio": json_numpy.dumps(proprio),
        "language_instruction": args.prompt,
        "image0": json_numpy.dumps(image_np),
        "domain_id": 0,
        "steps": 10,
    }

    try:
        payload_size_mb = len(payload["image0"].encode("utf-8")) / (1024 * 1024)
        print(f"Sending request to {args.url}")
        print(f"Image shape: {image_np.shape}, serialized image0 size: {payload_size_mb:.2f} MB")
        print(f"Timeouts => connect: {args.connect_timeout}s, read: {args.read_timeout}s")

        started = time.time()
        response = requests.post(
            args.url,
            json=payload,
            timeout=(args.connect_timeout, args.read_timeout),
        )
        elapsed = time.time() - started

        print(f"HTTP {response.status_code} in {elapsed:.2f}s")
        response.raise_for_status()

        result = response.json()
        action = np.asarray(result["action"], dtype=np.float32)
        print(f"Action length: {action.shape[0]}")
        print(f"Action: {action.tolist()}")

    except requests.exceptions.ConnectTimeout:
        print("Connect timeout: server is unreachable on this host/port.")
    except requests.exceptions.ReadTimeout:
        print("Read timeout: request reached server, but inference/model load took longer than read-timeout.")
        print("Tip: increase --read-timeout. First run can be slow while model loads.")
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response is not None else "<no response body>"
        print(f"HTTP error: {exc}")
        print(f"Response body: {body}")
    except Exception as exc:
        print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
