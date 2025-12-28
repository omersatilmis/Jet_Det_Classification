import { ModelData } from './app/components/hud-data';

// Interfaces matching the FastAPI Pydantic models
export interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface Detection {
    class_name: string;
    confidence: number;
    box: BoundingBox;
    azimuth?: number;
    elevation?: number;
    distance_km?: number;
}

export interface ModelMetrics {
    inference_time_ms: number;
    fps?: number;
    gpu_usage?: number;
    vram_usage_mb?: number;
}

export interface ModelResult {
    model_id: string;
    model_name: string;
    detections: Detection[];
    metrics: ModelMetrics;
    visualized_image?: string;
    heatmap_image?: string;
}

export interface EnsembleMetrics {
    consensus_score: number;
    iou_threshold: number;
    sigma: number;
}

export interface EnsembleResult {
    detections: Detection[];
    metrics: EnsembleMetrics;
}

export interface AnalysisResponse {
    status: string;
    image_id: string;
    models: ModelResult[];
    ensemble?: EnsembleResult;
}

/**
 * Sends an image to the FastAPI backend for analysis
 */
export const analyzeImage = async (file: File, activeModels: ModelData[]): Promise<AnalysisResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    // In the future, pass active models parameter here if needed
    // const modelIds = activeModels.map(m => m.id).join(',');
    // formData.append('models', modelIds);

    const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return await response.json() as AnalysisResponse;
};

/**
 * Uploads a new AI model to the backend
 */
export async function uploadModel(name: string, architecture: string, color: string, modelFile: File): Promise<any> {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('architecture', architecture);
    formData.append('color', color);
    formData.append('file', modelFile);

    const response = await fetch('/api/upload-model', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Model upload failed: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Analyzes an image using a single specified model via the `/api/analyze-single` endpoint.
 */
export async function analyzeSingleImage(modelId: string, imageFile: File): Promise<AnalysisResponse> {
    const formData = new FormData();
    formData.append('model_id', modelId);
    formData.append('file', imageFile);

    const response = await fetch('/api/analyze-single', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Single analysis failed: ${response.statusText}`);
    }

    return await response.json() as AnalysisResponse;
}

/**
 * Fetches the list of user-uploaded models from the persistent backend
 */
export async function fetchModels(): Promise<{ status: string, models: any[] }> {
    const response = await fetch('/api/models');
    if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
    }
    return response.json();
}

/**
 * Deletes a custom model from the backend
 */
export async function deleteModel(modelId: string): Promise<any> {
    const response = await fetch(`/api/models/${modelId}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error(`Failed to delete model: ${response.statusText}`);
    }
    return response.json();
}
