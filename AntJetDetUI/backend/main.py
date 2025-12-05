import os
import sys

# Critical: Prevent PyTorch DLL Deadlock on Windows ASGI
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure backend package structure is resolvable by Python
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import random
import asyncio

from inference import run_inference, init_cascade_rcnn_r50
from model_registry import load_models, save_models

import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Jet Aircraft Detection System API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err = traceback.format_exc()
    print(f"CRITICAL 500 ERROR:\n{err}")
    with open("backend_crash.log", "w", encoding="utf-8") as f:
        f.write(err)
    return JSONResponse(status_code=500, content={"message": str(exc), "traceback": err})

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

dynamic_models = load_models()

@app.on_event("startup")
async def startup_event():
    print("[INFO] Starting up API and preemptively loading AI models to VRAM...")
    # Lazy-load the cascade model so the first request doesn't lag 5+ seconds
    init_cascade_rcnn_r50()

# Allow CORS for local development with Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Detection(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance_km: Optional[float] = None

class ModelMetrics(BaseModel):
    inference_time_ms: float
    fps: Optional[float] = None
    gpu_usage: Optional[float] = None
    vram_usage_mb: Optional[float] = None

class ModelResult(BaseModel):
    model_id: str
    model_name: str
    detections: List[Detection]
    visualized_image: Optional[str] = None
    heatmap_image: Optional[str] = None
    metrics: ModelMetrics

class EnsembleMetrics(BaseModel):
    consensus_score: float
    iou_threshold: float
    sigma: float

class EnsembleResult(BaseModel):
    detections: List[Detection]
    metrics: EnsembleMetrics

class AnalysisResponse(BaseModel):
    status: str
    image_id: str
    models: List[ModelResult]
    ensemble: Optional[EnsembleResult] = None


# Mock Inference Functions - To be replaced by inference.py
async def mock_cascade_rcnn_inference() -> ModelResult:
    # Temporarily kept until inference.py is connected
    await asyncio.sleep(0.4) 
    return ModelResult(
        model_id="cascade-rcnn",
        model_name="Cascade R-CNN R50",
        detections=[],
        metrics=ModelMetrics(
            inference_time_ms=random.uniform(115.0, 135.0),
            gpu_usage=random.uniform(88.0, 95.0),
            vram_usage_mb=4520.0
        )
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...)
):
    """
    Deprecated for Independent Analysis. Will only run cascade-rcnn if called.
    """
    print(f"[INFO] Received file for analysis: {file.filename}")
    start_time = time.time()
    
    model_result = await mock_cascade_rcnn_inference()
    
    total_time = (time.time() - start_time) * 1000
    print(f"[INFO] Analysis complete in {total_time:.2f}ms")

    return AnalysisResponse(
        status="success",
        image_id=f"img_{int(time.time())}",
        models=[model_result],
        ensemble=None
    )

@app.post("/api/analyze-single", response_model=AnalysisResponse)
async def analyze_single(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Analyzes an uploaded image using a SINGLE specified model.
    """
    print(f"[INFO] Received file for SINGLE analysis: {file.filename} with model: {model_id}")
    start_time = time.time()
    
    # Read the image bytes
    content = await file.read()
    
    with open("debug_analyze.log", "a") as dbg: dbg.write(f"\n[INFO] Starting analysis for {model_id}\n")
    
    model_result = None
    if model_id == "cascade-rcnn":
        # 1. Run real PyTorch inference
        try:
            inference_out = await asyncio.to_thread(run_inference, content)
            with open("debug_analyze.log", "a") as dbg: dbg.write(f"Inference Out: {type(inference_out)} - keys: {inference_out.keys()}\n")
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            with open("debug_analyze.log", "a") as dbg: dbg.write(f"CRASH IN RUN_INFERENCE:\n{err}\n")
            raise e
            
        # 2. Map inference output to standard ModelResult
        if inference_out.get("success"):
            metrics_dict = inference_out.get("metrics", {})
            detections = []
            for det in inference_out.get("detections", []):
                detections.append(Detection(
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    box=BoundingBox(
                        x=det["box"]["x"],
                        y=det["box"]["y"],
                        width=det["box"]["width"],
                        height=det["box"]["height"]
                    )
                ))
            try:
                model_result = ModelResult(
                    model_id=model_id,
                    model_name="Cascade R-CNN R50",
                    detections=detections,
                    visualized_image=inference_out.get("visualized_image"),
                    heatmap_image=inference_out.get("heatmap_image"),
                    metrics=ModelMetrics(
                        inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                        fps=metrics_dict.get("fps", 0.0),
                        gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                        vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0)
                    )
                )
                with open("debug_analyze.log", "a") as dbg: dbg.write("ModelResult created successfully\n")
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                with open("debug_analyze.log", "a") as dbg: dbg.write(f"CRASH IN PYDANTIC VALIDATION:\n{err}\n")
                raise e
        else:
             print(f"[ERROR] Inference failed internal to script: {inference_out.get('error')}")
             model_result = ModelResult(
                 model_id=model_id,
                 model_name="Cascade R-CNN R50 [FAILED]",
                 detections=[],
                 metrics=ModelMetrics(inference_time_ms=0.0)
             )
    elif model_id in dynamic_models:
        # Run real PyTorch inference using the custom uploaded .pth file
        inference_out = await asyncio.to_thread(run_inference, content, model_id, dynamic_models)
        
        if inference_out.get("success"):
            metrics_dict = inference_out.get("metrics", {})
            detections = []
            for det in inference_out.get("detections", []):
                detections.append(Detection(
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    box=BoundingBox(
                        x=det["box"]["x"],
                        y=det["box"]["y"],
                        width=det["box"]["width"],
                        height=det["box"]["height"]
                    )
                ))
            model_result = ModelResult(
                model_id=model_id,
                model_name=dynamic_models[model_id]["name"],
                detections=detections,
                visualized_image=inference_out.get("visualized_image"),
                heatmap_image=inference_out.get("heatmap_image"),
                metrics=ModelMetrics(
                    inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                    fps=metrics_dict.get("fps", 0.0),
                    gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                    vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0)
                )
            )
        else:
             print(f"[ERROR] Inference failed for custom model: {inference_out.get('error')}")
             model_result = ModelResult(
                 model_id=model_id,
                 model_name=f"{dynamic_models[model_id]['name']} [FAILED]",
                 detections=[],
                 metrics=ModelMetrics(inference_time_ms=0.0)
             )
    else:
        model_result = ModelResult(
            model_id=model_id,
            model_name=f"Unknown Model",
            detections=[],
            metrics=ModelMetrics(inference_time_ms=0.0)
        )
         
    total_time = (time.time() - start_time) * 1000
    print(f"[INFO] Single Analysis complete in {total_time:.2f}ms")

    return AnalysisResponse(
        status="success",
        image_id=f"img_{int(time.time())}",
        models=[model_result] if model_result else [],
        ensemble=None
    )

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "System Operational"}

# Mock Registry for Dynamic Models (Now Persistent)
# dynamic_models is loaded at the top of the file

@app.post("/api/upload-model")
async def upload_model(
    name: str = Form(...),
    architecture: str = Form(...),
    color: str = Form(...),
    file: UploadFile = File(...)
):
    print(f"[INFO] Uploading new user model: {name} ({architecture})")
    content = await file.read()
    model_path = os.path.join(MODELS_DIR, file.filename)
    
    with open(model_path, "wb") as f:
        f.write(content)
    
    # Simulating a small delay
    await asyncio.sleep(0.5)
    
    model_id = f"custom-{name.lower().replace(' ', '-')}"
    
    # Add to dynamic registry
    new_model = {
        "id": model_id,
        "name": name,
        "shortName": name[:8].upper(),
        "color": color, 
        "architecture": architecture,
        "status": "Ready",
        "file_path": model_path
    }
    dynamic_models[model_id] = new_model
    save_models(dynamic_models)
    
    return {
        "status": "success",
        "message": f"Model '{name}' registered successfully.",
        "model": new_model
    }

@app.get("/api/models")
async def get_models():
    """Return the list of custom uploaded models from the persistent registry"""
    return {"status": "success", "models": list(dynamic_models.values())}

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Deletes a custom model from the registry and removes its physical file"""
    if model_id not in dynamic_models:
        return {"status": "error", "message": "Model not found"}
        
    model_data = dynamic_models[model_id]
    
    # Try to remove the physical file
    file_path = model_data.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"[INFO] Deleted model file: {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not delete file {file_path}: {e}")
            
    # Remove from registry and save
    del dynamic_models[model_id]
    save_models(dynamic_models)
    
    return {"status": "success", "message": f"Model {model_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
