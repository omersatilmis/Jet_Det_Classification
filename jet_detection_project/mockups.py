import os
import time

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def wait():
    input("\nPress ENTER to continue...")
    clear()


# ------------------- MAIN MENU -------------------
def screen_main_menu():
    clear()
    print("=========================================")
    print("     Jet Detection & Classification")
    print("=========================================\n")
    print("1) Analyze Dataset")
    print("2) Train Model")
    print("3) Run Inference on Image")
    print("4) Run Inference on Folder")
    print("5) Run Inference on Video")
    print("6) Exit\n")
    input("Select an option: _")
    wait()


# ---------------- DATASET ANALYSIS ----------------
def screen_dataset_analysis():
    print("---------------- Dataset Analysis ----------------\n")
    print("Total Images: 4,812")
    print("Total Annotations: 6,249")
    print("Number of Classes: 80\n")
    print("Top 5 Most Frequent Jet Models:")
    print("1. F-16 (892)")
    print("2. F-35 (741)")
    print("3. F-22 (650)")
    print("4. SU-27 (489)")
    print("5. MIG-29 (402)\n")
    print("Bounding Box Statistics:")
    print("- Average Width: 142.3 px")
    print("- Average Height: 126.9 px\n")
    print("Plots saved to: /outputs/dataset_analysis/\n")
    print("--------------------------------------------------")
    wait()


# ------------------ TRAINING SCREEN ------------------
def screen_training():
    print("---------------- Model Training ----------------\n")
    print("Config file: cascade_rcnn_swin_tiny.py")
    print("Epochs: 20")
    print("Batch size: 2")
    print("Learning rate: 0.0001\n")
    print("Training started...")

    epochs = [1, 2, 3]
    losses = [1.84, 1.61, 1.47]

    for i, loss in zip(epochs, losses):
        time.sleep(0.7)
        print(f"[Epoch {i}/20] Loss: {loss}   LR: 1e-4")

    print("...\nBest checkpoint saved at: /checkpoints/epoch_14.pth\n")
    print("Training completed successfully!")
    wait()


# ------------------ IMAGE INFERENCE ------------------
def screen_inference_image():
    print("---------------- Inference: Single Image ----------------\n")
    print("Enter image path: input_images/test_22.jpg\n")
    print("Running detection...")
    time.sleep(1)
    print("Detected 2 jet(s):\n")
    print("1) Class: F-22  | Confidence: 0.94 | BBox: [242, 119, 610, 448]")
    print("2) Class: F-35  | Confidence: 0.88 | BBox: [701, 145, 961, 470]\n")
    print("Output image saved to: /outputs/inference/test_22_result.jpg")
    print("JSON results saved to: /outputs/inference/test_22.json")
    wait()


# --------------------- VIDEO INFERENCE ---------------------
def screen_inference_video():
    print("---------------- Inference: Video ----------------\n")
    print("Enter video path: videos/f35_chase.mp4")
    print("Output video: outputs/video_results/f35_chase_out.mp4\n")

    for i in range(1, 4):
        time.sleep(0.7)
        print(f"Processing frame {i}/936...")

    print("...\n")
    time.sleep(1)
    print("Video completed.")
    print("Results saved successfully.")
    wait()


# --------------------- RUN ALL SCREENS ---------------------
def run_demo():
    screen_main_menu()
    screen_dataset_analysis()
    screen_training()
    screen_inference_image()
    screen_inference_video()
    print("Demo Finished.")


if __name__ == "__main__":
    run_demo()
