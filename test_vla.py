import argparse
import time

import json_numpy
import numpy as np
import requests
from PIL import Image


def post_xvla(url, image_np, prompt, connect_timeout, read_timeout):
    payload = {
        "language_instruction": prompt,
        "image0": json_numpy.dumps(image_np),
        "domain_id": 0,
        "steps": 10,
    }

    payload_size_mb = len(payload["image0"].encode("utf-8")) / (1024 * 1024)
    print(f"Image shape: {image_np.shape}, serialized image0 size: {payload_size_mb:.2f} MB")

    return requests.post(
        url,
        json=payload,
        timeout=(connect_timeout, read_timeout),
    )


def post_xvla_with_optional_proprio(url, image_np, prompt, connect_timeout, read_timeout, include_proprio, proprio_dim):
    if not include_proprio:
        return post_xvla(url, image_np, prompt, connect_timeout, read_timeout)

    proprio = np.zeros(proprio_dim, dtype=np.float32)
    payload = {
        "proprio": json_numpy.dumps(proprio),
        "language_instruction": prompt,
        "image0": json_numpy.dumps(image_np),
        "domain_id": 0,
        "steps": 10,
    }

    payload_size_mb = len(payload["image0"].encode("utf-8")) / (1024 * 1024)
    print(f"Image shape: {image_np.shape}, serialized image0 size: {payload_size_mb:.2f} MB")

    return requests.post(
        url,
        json=payload,
        timeout=(connect_timeout, read_timeout),
    )


def post_fastapi(url, image_path, prompt, connect_timeout, read_timeout):
    with open(image_path, "rb") as image_file:
        files = {"image": (image_path, image_file, "application/octet-stream")}
        data = {"text": prompt}
        return requests.post(
            url,
            data=data,
            files=files,
            timeout=(connect_timeout, read_timeout),
        )


def looks_like_fastapi_422(response):
    if response.status_code != 422:
        return False
    body = response.text
    return '"body","text"' in body and '"body","image"' in body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9090/act")
    parser.add_argument("--image", default="lego_on_table.png")
    parser.add_argument("--prompt", default="pick up blue block")
    parser.add_argument(
        "--interface",
        choices=["auto", "xvla", "fastapi"],
        default="auto",
        help="Request format: XVLA JSON, FastAPI multipart, or auto-detect",
    )
    parser.add_argument(
        "--include-proprio",
        action="store_true",
        help="Include proprio vector in XVLA payload (disabled by default for compatibility)",
    )
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)

    try:
        print(f"Sending request to {args.url}")
        print(f"Interface mode: {args.interface}")
        print(f"Timeouts => connect: {args.connect_timeout}s, read: {args.read_timeout}s")

        started = time.time()
        if args.interface == "xvla":
            response = post_xvla_with_optional_proprio(
                args.url,
                image_np,
                args.prompt,
                args.connect_timeout,
                args.read_timeout,
                args.include_proprio,
                args.proprio_dim,
            )
        elif args.interface == "fastapi":
            response = post_fastapi(args.url, args.image, args.prompt, args.connect_timeout, args.read_timeout)
        else:
            response = post_xvla_with_optional_proprio(
                args.url,
                image_np,
                args.prompt,
                args.connect_timeout,
                args.read_timeout,
                args.include_proprio,
                args.proprio_dim,
            )
            if looks_like_fastapi_422(response):
                print("Endpoint expects FastAPI form-data (text + image). Retrying automatically...")
                response = post_fastapi(args.url, args.image, args.prompt, args.connect_timeout, args.read_timeout)

        elapsed = time.time() - started

        print(f"HTTP {response.status_code} in {elapsed:.2f}s")
        response.raise_for_status()

        result = response.json()
        if "action" in result:
            action = np.asarray(result["action"], dtype=np.float32)
            print(f"Action length: {action.shape[0]}")
            print(f"Action: {action.tolist()}")
        elif "full_action_vector" in result:
            action = np.asarray(result["full_action_vector"], dtype=np.float32)
            print(f"Action length: {action.shape[0]}")
            print(f"Action: {action.tolist()}")
            if "interpreted_action" in result:
                print(f"Interpreted action: {result['interpreted_action']}")
        else:
            print(f"Response JSON: {result}")

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
