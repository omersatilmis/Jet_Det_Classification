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

from inference import run_inference, run_video_inference, init_cascade_rcnn_r50
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

EVAL_METRICS_FILE = "eval_metrics.json"
_eval_metrics_cache = {}
_eval_metrics_mtime = 0.0

def load_eval_metrics():
    if not os.path.exists(EVAL_METRICS_FILE):
        return {}
    try:
        import json
        with open(EVAL_METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {EVAL_METRICS_FILE}: {e}")
        return {}

def get_eval_metrics():
    global _eval_metrics_cache, _eval_metrics_mtime
    try:
        mtime = os.path.getmtime(EVAL_METRICS_FILE)
    except Exception:
        mtime = 0.0

    if mtime != _eval_metrics_mtime:
        _eval_metrics_cache = load_eval_metrics()
        _eval_metrics_mtime = mtime
    return _eval_metrics_cache

@app.on_event("startup")
async def startup_event():
    print("[INFO] Starting up API and preemptively loading AI models to VRAM...")
    # Lazy-load the cascade model so the first request doesn't lag 5+ seconds
    init_cascade_rcnn_r50()

# Allow CORS for local development with Vite
cors_env = os.environ.get("CORS_ORIGINS")
if cors_env:
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
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
    map: Optional[float] = None
    iou: Optional[float] = None


class PRPoint(BaseModel):
    recall: float
    precision: float


class VideoFrameDetections(BaseModel):
    timestamp_ms: float
    detections: List[Detection]

class ModelResult(BaseModel):
    model_id: str
    model_name: str
    detections: List[Detection]
    visualized_image: Optional[str] = None
    heatmap_image: Optional[str] = None
    metrics: ModelMetrics
    frame_detections: Optional[List[VideoFrameDetections]] = None
    pr_curve: Optional[List[PRPoint]] = None

class EnsembleMetrics(BaseModel):
    consensus_score: float
    iou_threshold: float
    sigma: float
    avg_inference_time_ms: Optional[float] = None
    avg_gpu_usage: Optional[float] = None
    avg_vram_usage_mb: Optional[float] = None
    avg_fps: Optional[float] = None
    avg_map: Optional[float] = None
    avg_iou: Optional[float] = None

class EnsembleResult(BaseModel):
    detections: List[Detection]
    metrics: EnsembleMetrics

class AnalysisResponse(BaseModel):
    status: str
    image_id: str
    models: List[ModelResult]
    ensemble: Optional[EnsembleResult] = None

class EnsembleRequest(BaseModel):
    results: List[ModelResult]

@app.post("/api/compute-ensemble", response_model=EnsembleResult)
async def compute_ensemble(request: EnsembleRequest):
    """
    Computes ensemble (WBF) results from a list of previously computed model results.
    """
    from ensemble_logic import weighted_box_fusion, calculate_ensemble_metrics
    
    if not request.results:
        return EnsembleResult(
            detections=[],
            metrics=EnsembleMetrics(consensus_score=0, iou_threshold=0.5, sigma=0.0)
        )
        
    detections_by_model = [r.detections for r in request.results]
    # Convert Pydantic models back to dict for the logic module
    detections_dicts = []
    for model_dets in detections_by_model:
        detections_dicts.append([d.dict() for d in model_dets])
        
    fused_detections = weighted_box_fusion(detections_dicts)
    metrics = calculate_ensemble_metrics([r.dict() for r in request.results])
    
    return EnsembleResult(
        detections=[Detection(**d) for d in fused_detections],
        metrics=EnsembleMetrics(
            consensus_score=sum(d['agreement'] for d in fused_detections)/max(len(fused_detections), 1) if fused_detections else 1.0,
            iou_threshold=0.5,
            sigma=0.0,
            avg_inference_time_ms=metrics.get("avg_inference_time_ms"),
            avg_gpu_usage=metrics.get("avg_gpu_usage"),
            avg_vram_usage_mb=metrics.get("avg_vram_usage_mb"),
            avg_fps=metrics.get("avg_fps"),
            avg_map=metrics.get("avg_map"),
            avg_iou=metrics.get("avg_iou")
        )
    )






@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    models: Optional[str] = Form(None) # Comma separated model IDs
):
    """
    Analyzes an uploaded image using MULTIPLE models and computes ensemble results.
    """
    print(f"[INFO] Received file for MULTI-MODEL analysis: {file.filename}")
    start_time = time.time()
    
    # 1. Parse model IDs
    requested_model_ids = ["cascade-rcnn-r50-tiny"] # Default
    if models:
        requested_model_ids = [m.strip() for m in models.split(",") if m.strip()]
    
    # 2. Read file content once
    content = await file.read()
    
    # 3. Import logic
    from ensemble_logic import weighted_box_fusion, calculate_ensemble_metrics
    
    results = []
    detections_by_model = []
    
    # 4. Run each model (Sequential for now to avoid VRAM overload, can be parallelized if small)
    for mid in requested_model_ids:
        try:
            # We reuse the logic from analyze_single but as a helper
            inference_out = await asyncio.to_thread(run_inference, content, mid, dynamic_models if mid in dynamic_models else None)
            
            if inference_out.get("success"):
                metrics_dict = inference_out.get("metrics", {})
                detections = []
                for det in inference_out.get("detections", []):
                    detections.append(Detection(**det))
                
                model_name = "Cascade R-CNN R50 Tiny"
                if mid in dynamic_models:
                    model_name = dynamic_models[mid]["name"]
                
                eval_data = get_eval_metrics().get(mid, {})
                frame_dets = [VideoFrameDetections(timestamp_ms=f.get("timestamp_ms", 0.0),
                                                  detections=[Detection(**d) for d in f.get("detections", [])])
                              for f in inference_out.get("frame_detections", [])]
                pr_curve = [PRPoint(**p) for p in inference_out.get("pr_curve", [])]
                if not pr_curve:
                    pr_curve = [PRPoint(**p) for p in (eval_data.get("pr_curve") or [])]
                
                model_result = ModelResult(
                    model_id=mid,
                    model_name=model_name,
                    detections=detections,
                    visualized_image=inference_out.get("visualized_image"),
                    heatmap_image=inference_out.get("heatmap_image"),
                    frame_detections=frame_dets,
                    pr_curve=pr_curve,
                    metrics=ModelMetrics(
                        inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                        fps=metrics_dict.get("fps", 0.0),
                        gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                        vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0),
                        map=metrics_dict.get("map") or eval_data.get("map"),
                        iou=metrics_dict.get("iou") or eval_data.get("iou")
                    )
                )
                results.append(model_result)
                detections_by_model.append(inference_out.get("detections", []))
            else:
                print(f"[ERROR] Inference failed for model {mid}: {inference_out.get('error')}")
        except Exception as e:
            print(f"[ERROR] Exception running model {mid}: {e}")
            
    # 5. Compute Ensemble
    ensemble_result = None
    if len(results) > 0:
        fused_detections = weighted_box_fusion(detections_by_model)
        metrics = calculate_ensemble_metrics([r.dict() for r in results])
        
        ensemble_result = EnsembleResult(
            detections=[Detection(**d) for d in fused_detections],
            metrics=EnsembleMetrics(
                consensus_score=sum(d['agreement'] for d in fused_detections)/max(len(fused_detections), 1) if fused_detections else 1.0,
                iou_threshold=0.5,
                sigma=0.0, # Not used for now
                avg_inference_time_ms=metrics.get("avg_inference_time_ms"),
                avg_gpu_usage=metrics.get("avg_gpu_usage"),
                avg_vram_usage_mb=metrics.get("avg_vram_usage_mb"),
                avg_fps=metrics.get("avg_fps"),
                avg_map=metrics.get("avg_map"),
                avg_iou=metrics.get("avg_iou")
            )
        )
        # Update metrics with avg values
        # metrics["consensus_score"] = ensemble_result.metrics.consensus_score

    total_time = (time.time() - start_time) * 1000
    print(f"[INFO] Multi-Model Analysis complete in {total_time:.2f}ms")

    return AnalysisResponse(
        status="success",
        image_id=f"img_{int(time.time())}",
        models=results,
        ensemble=ensemble_result
    )


@app.post("/api/analyze-video", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    models: Optional[str] = Form(None)
):
    """
    Analyzes an uploaded video using MULTIPLE models and computes ensemble results.
    """
    print(f"[INFO] Received file for MULTI-MODEL video analysis: {file.filename}")
    start_time = time.time()

    requested_model_ids = ["cascade-rcnn-r50-tiny"]
    if models:
        requested_model_ids = [m.strip() for m in models.split(",") if m.strip()]

    content = await file.read()

    from ensemble_logic import weighted_box_fusion, calculate_ensemble_metrics

    results = []
    detections_by_model = []

    for mid in requested_model_ids:
        try:
            inference_out = await asyncio.to_thread(
                run_video_inference,
                content,
                mid,
                dynamic_models if mid in dynamic_models else None,
                file.filename,
            )

            if inference_out.get("success"):
                metrics_dict = inference_out.get("metrics", {})
                detections = [Detection(**det) for det in inference_out.get("detections", [])]

                model_name = "Cascade R-CNN R50 Tiny"
                if mid in dynamic_models:
                    model_name = dynamic_models[mid]["name"]

                eval_data = get_eval_metrics().get(mid, {})
                frame_dets = [VideoFrameDetections(timestamp_ms=f.get("timestamp_ms", 0.0),
                                                  detections=[Detection(**d) for d in f.get("detections", [])])
                              for f in inference_out.get("frame_detections", [])]
                pr_curve = [PRPoint(**p) for p in inference_out.get("pr_curve", [])]
                if not pr_curve:
                    pr_curve = [PRPoint(**p) for p in (eval_data.get("pr_curve") or [])]
                
                model_result = ModelResult(
                    model_id=mid,
                    model_name=model_name,
                    detections=detections,
                    visualized_image=inference_out.get("visualized_image"),
                    heatmap_image=inference_out.get("heatmap_image"),
                    frame_detections=frame_dets,
                    pr_curve=pr_curve,
                    metrics=ModelMetrics(
                        inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                        fps=metrics_dict.get("fps", 0.0),
                        gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                        vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0),
                        map=metrics_dict.get("map") or eval_data.get("map"),
                        iou=metrics_dict.get("iou") or eval_data.get("iou"),
                    )
                )
                results.append(model_result)
                detections_by_model.append(inference_out.get("detections", []))
            else:
                print(f"[ERROR] Video inference failed for model {mid}: {inference_out.get('error')}")
        except Exception as e:
            print(f"[ERROR] Exception running video model {mid}: {e}")

    ensemble_result = None
    if len(results) > 0:
        fused_detections = weighted_box_fusion(detections_by_model)
        metrics = calculate_ensemble_metrics([r.dict() for r in results])

        ensemble_result = EnsembleResult(
            detections=[Detection(**d) for d in fused_detections],
            metrics=EnsembleMetrics(
                consensus_score=sum(d['agreement'] for d in fused_detections)/max(len(fused_detections), 1) if fused_detections else 1.0,
                iou_threshold=0.5,
                sigma=0.0,
                avg_inference_time_ms=metrics.get("avg_inference_time_ms"),
                avg_gpu_usage=metrics.get("avg_gpu_usage"),
                avg_vram_usage_mb=metrics.get("avg_vram_usage_mb"),
                avg_fps=metrics.get("avg_fps"),
                avg_map=metrics.get("avg_map"),
                avg_iou=metrics.get("avg_iou")
            )
        )

    total_time = (time.time() - start_time) * 1000
    print(f"[INFO] Multi-Model Video Analysis complete in {total_time:.2f}ms")

    return AnalysisResponse(
        status="success",
        image_id=f"vid_{int(time.time())}",
        models=results,
        ensemble=ensemble_result
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
    if model_id == "cascade-rcnn-r50-tiny":
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
            detections = [Detection(**det) for det in inference_out.get("detections", [])]
            try:
                eval_data = get_eval_metrics().get(model_id, {})
                pr_curve = [PRPoint(**p) for p in inference_out.get("pr_curve", [])]
                if not pr_curve:
                    pr_curve = [PRPoint(**p) for p in (eval_data.get("pr_curve") or [])]
                    
                model_result = ModelResult(
                    model_id=model_id,
                    model_name="Cascade R-CNN R50 Tiny",
                    detections=detections,
                    visualized_image=inference_out.get("visualized_image"),
                    heatmap_image=inference_out.get("heatmap_image"),
                    pr_curve=pr_curve,
                    metrics=ModelMetrics(
                        inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                        fps=metrics_dict.get("fps", 0.0),
                        gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                        vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0),
                        map=metrics_dict.get("map") or eval_data.get("map"),
                        iou=metrics_dict.get("iou") or eval_data.get("iou")
                    )
                )
                print(f"[DEBUG] Single Model PR Curve length: {len(pr_curve)}")
                print(f"[DEBUG] Single Model mAP: {model_result.metrics.map}")
                model_result = model_result
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
            detections = [Detection(**det) for det in inference_out.get("detections", [])]
            eval_data = get_eval_metrics().get(model_id, {})
            pr_curve = [PRPoint(**p) for p in inference_out.get("pr_curve", [])]
            if not pr_curve:
                pr_curve = [PRPoint(**p) for p in (eval_data.get("pr_curve") or [])]
                
            model_result = ModelResult(
                model_id=model_id,
                model_name=dynamic_models[model_id]["name"],
                detections=detections,
                visualized_image=inference_out.get("visualized_image"),
                heatmap_image=inference_out.get("heatmap_image"),
                pr_curve=pr_curve,
                metrics=ModelMetrics(
                    inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                    fps=metrics_dict.get("fps", 0.0),
                    gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                    vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0),
                    map=metrics_dict.get("map") or eval_data.get("map"),
                    iou=metrics_dict.get("iou") or eval_data.get("iou")
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


@app.post("/api/analyze-video-single", response_model=AnalysisResponse)
async def analyze_video_single(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Analyzes an uploaded video using a SINGLE specified model.
    """
    print(f"[INFO] Received file for SINGLE video analysis: {file.filename} with model: {model_id}")
    start_time = time.time()

    content = await file.read()

    model_result = None
    if model_id == "cascade-rcnn-r50-tiny" or model_id in dynamic_models:
        registry = dynamic_models if model_id in dynamic_models else None
        inference_out = await asyncio.to_thread(run_video_inference, content, model_id, registry, file.filename)

        if inference_out.get("success"):
            metrics_dict = inference_out.get("metrics", {})
            detections = [Detection(**det) for det in inference_out.get("detections", [])]

            model_name = "Cascade R-CNN R50 Tiny"
            if model_id in dynamic_models:
                model_name = dynamic_models[model_id]["name"]

            eval_data = get_eval_metrics().get(model_id, {})
            frame_dets = [VideoFrameDetections(timestamp_ms=f.get("timestamp_ms", 0.0),
                                              detections=[Detection(**d) for d in f.get("detections", [])])
                          for f in inference_out.get("frame_detections", [])]
            pr_curve = [PRPoint(**p) for p in inference_out.get("pr_curve", [])]
            if not pr_curve:
                pr_curve = [PRPoint(**p) for p in (eval_data.get("pr_curve") or [])]
                
            model_result = ModelResult(
                model_id=model_id,
                model_name=model_name,
                detections=detections,
                visualized_image=inference_out.get("visualized_image"),
                heatmap_image=inference_out.get("heatmap_image"),
                frame_detections=frame_dets,
                pr_curve=pr_curve,
                metrics=ModelMetrics(
                    inference_time_ms=metrics_dict.get("inference_time_ms", 0.0),
                    fps=metrics_dict.get("fps", 0.0),
                    gpu_usage=metrics_dict.get("gpu_usage", 0.0),
                    vram_usage_mb=metrics_dict.get("vram_usage_mb", 0.0),
                    map=metrics_dict.get("map") or eval_data.get("map"),
                    iou=metrics_dict.get("iou") or eval_data.get("iou"),
                )
            )
        else:
            print(f"[ERROR] Video inference failed for model {model_id}: {inference_out.get('error')}")

    total_time = (time.time() - start_time) * 1000
    print(f"[INFO] Single Video Analysis complete in {total_time:.2f}ms")

    return AnalysisResponse(
        status="success",
        image_id=f"vid_{int(time.time())}",
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
