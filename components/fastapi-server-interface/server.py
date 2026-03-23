import argparse
import contextlib
import io
import logging
import os
import traceback
import torch
import uvicorn
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", ""))
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
parser.add_argument("--unnorm_key", type=str, default=os.environ.get("UNNORM_KEY", "bridge_orig"))
args, _ = parser.parse_known_args()

model = None
processor = None
model_ready = False


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ready")
def ready():
    if model_ready:
        return {"ready": True}
    return Response(status_code=503, content='{"ready": false}', media_type="application/json")


@app.post("/act")
async def predict_action(text: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"In: {text}\nOut:"
        inputs = processor(prompt, img).to(args.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

        return JSONResponse({
            "full_action_vector": action.tolist(),
            "gripper_pose": float(action[-1]),
            "interpreted_action": "GRAB" if action[-1] > 0.5 else "RELEASE",
        })
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Internal server error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
