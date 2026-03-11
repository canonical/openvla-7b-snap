import argparse
import torch
import uvicorn
import io
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

app = FastAPI()
model = None
processor = None
model_ready = False

# 1. Start-up: Load the 15GB model into RAM once
@app.on_event("startup")
def load_vla():
    global model, processor, model_ready
    print(f"🚀 Loading OpenVLA from {args.model_path} on {args.device}...")

    # trust_remote_code is required for the Prismatic architecture
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device)
    model_ready = True
    print("✅ Model loaded and ready for actions.")


@app.get("/ready")
def ready():
    if model_ready:
        return {"ready": True}
    from fastapi import Response
    return Response(status_code=503, content='{"ready": false}', media_type="application/json")

# 2. The Inference Endpoint
@app.post("/act")
async def predict_action(text: str = Form(...), image: UploadFile = File(...)):
    # Convert uploaded bytes to a PIL Image
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Format the prompt for OpenVLA
    prompt = f"In: {text}\nOut:"

    # Prepare inputs
    inputs = processor(prompt, img).to(args.device, dtype=torch.bfloat16)

    # Predict the action (X, Y, Z, Roll, Pitch, Yaw, Gripper)
    # unnorm_key="bridge_orig" is the standard for the Bridge dataset OpenVLA was trained on
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Return the gripper pose (the 7th number) and the full vector
    # 1.0 = Closed/Grab, 0.0 = Open
    return {
        "full_action_vector": action.tolist(),
        "gripper_pose": float(action[-1]),
        "interpreted_action": "GRAB" if action[-1] > 0.5 else "RELEASE"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
