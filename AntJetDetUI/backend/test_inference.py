import asyncio
from inference import run_inference
import urllib.request
import cv2
import numpy as np

async def main():
    # Download a test image
    url = "https://images.unsplash.com/photo-1750526997059-7ad806d66411?w=800"
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image_bytes = cv2.imencode('.jpg', cv2.imdecode(arr, -1))[1].tobytes()

    res = await run_inference(image_bytes)
    print("Success:", res["success"])
    if res["success"]:
        print("Detections:", res["detections"])
        print("Has visualized_image:", "visualized_image" in res)
        if "visualized_image" in res:
            print("Prefix:", res["visualized_image"][:50])

if __name__ == "__main__":
    asyncio.run(main())
