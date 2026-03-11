import argparse
import io
import json_numpy
import logging
import numpy as np
import traceback
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


app = FastAPI()
model = None
processor = None
model_ready = False


@app.on_event("startup")
def load_vla():
    global model, processor, model_ready
    print(f"🚀 Loading OpenVLA from {args.model_path} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model_ready = True
    print("✅ Model loaded and ready for actions.")


@app.get("/ready")
def ready():
    if model_ready:
        return {"ready": True}
    from fastapi import Response
    return Response(status_code=503, content='{"ready": false}', media_type="application/json")


def deserialize_image_payload(image_payload):
    value = json_numpy.loads(image_payload) if isinstance(image_payload, str) else image_payload

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            try:
                return Image.open(io.BytesIO(value.astype(np.uint8).tobytes())).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Unable to decode image bytes: {exc}") from exc

        image_array = value
    elif isinstance(value, list):
        image_array = np.asarray(value)
    else:
        raise ValueError("Image payload must deserialize to numpy array or list")

    if image_array.ndim not in (2, 3):
        raise ValueError("Image payload must be 2D or 3D")

    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating) and image_array.size > 0 and image_array.max() <= 1.0:
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array[:, :, 0]
    if image_array.ndim == 3 and image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]

    return Image.fromarray(image_array).convert("RGB")


@app.post("/act")
def predict_action(payload: dict):
    try:
        if "language_instruction" not in payload:
            return JSONResponse({"error": "Missing field: language_instruction"}, status_code=400)

        images = []
        for key in ("image0", "image1", "image2"):
            if key not in payload:
                continue
            images.append(deserialize_image_payload(payload[key]))

        if not images:
            return JSONResponse({"error": "No image provided. Include at least image0."}, status_code=400)

        proprio_payload = payload.get("proprio")
        if proprio_payload is not None:
            proprio = torch.as_tensor(np.asarray(json_numpy.loads(proprio_payload)))
        else:
            proprio = torch.zeros(7, dtype=torch.float32)

        domain_id = torch.tensor([int(payload.get("domain_id", 0))], dtype=torch.long)
        steps = max(1, int(payload.get("steps", 10)))

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def to_model(tensor_value):
            if not isinstance(tensor_value, torch.Tensor):
                tensor_value = torch.as_tensor(tensor_value)
            return tensor_value.to(device=device, dtype=dtype) if tensor_value.is_floating_point() else tensor_value.to(device=device)

        prompt = f"In: {payload['language_instruction']}\nOut:"
        inputs = processor(prompt, images[0])
        inputs = {key: to_model(value) for key, value in inputs.items()}

        _ = to_model(proprio.unsqueeze(0))
        _ = domain_id.to(device)
        _ = steps

        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        return JSONResponse({"action": action.float().cpu().numpy().tolist()})
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Request failed"}, status_code=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
