import asyncio
from backend.inference import run_inference
import sys

with open(r'c:\Users\omerf\Desktop\Jet_Det_Project\AntJetDetUI\public\vite.svg', 'rb') as f:
    b = f.read()

async def main():
    try:
        out = await run_inference(image_bytes=b, model_id='cascade-rcnn', custom_models_registry=None)
        print('SUCCESS:', out.keys())
    except Exception as e:
        import traceback
        traceback.print_exc()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

asyncio.run(main())
