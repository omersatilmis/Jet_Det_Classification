import sys
import os
backend_path = r"c:\Users\omerf\Desktop\Jet_Sentinel_Project\AntJetDetUI\backend"
sys.path.append(backend_path)

from ensemble_logic import weighted_box_fusion

def test_wbf():
    # Model 1: Close to center
    model1_dets = [
        {"class_name": "jet", "confidence": 0.9, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}}
    ]
    # Model 2: Slightly offset but overlapping (IoU > 0.5)
    model2_dets = [
        {"class_name": "jet", "confidence": 0.8, "box": {"x": 0.11, "y": 0.11, "width": 0.2, "height": 0.2}}
    ]
    # Model 3: Far away, shouldn't fuse
    model3_dets = [
        {"class_name": "jet", "confidence": 0.7, "box": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1}}
    ]

    detections_by_model = [model1_dets, model2_dets, model3_dets]
    results = weighted_box_fusion(detections_by_model, iou_threshold=0.5)

    print(f"Fused Results Count: {len(results)}")
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res['class_name']} conf: {res['confidence']:.2f} agreement: {res['agreement']:.2f}")

    assert len(results) == 2
    assert results[0]['model_count'] == 2
    assert results[1]['model_count'] == 1
    print("Test Passed!")

if __name__ == "__main__":
    test_wbf()
