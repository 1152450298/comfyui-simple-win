import gradio as gr
import requests
import uuid
import time
from PIL import Image
from io import BytesIO

def greet(prompt_int):
    comfy_submit_api = 'http://127.0.0.1:8188/api/prompt'
    id = uuid.uuid4()
    prompt = {"3": {
    "inputs": {
      "seed": 382780770644265,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "K采样器"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Checkpoint加载器（简易）"
    }
  },
  "5": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "空Latent图像"
    }
  },
  "6": {
    "inputs": {
      "text": prompt_int,
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP文本编码"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP文本编码"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE解码"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "保存图像"
    }
  }
}
    data = {
        "client_id" : str(id),
        "prompt" : prompt
    }
    p_id = requests.post(comfy_submit_api, json=data)
    p_id.raise_for_status()
    prompt_value = p_id.json()
    prompt_id = prompt_value.get("prompt_id")
    queue_url = 'http://127.0.0.1:8188/api/queue'
    while True:
      queue_info_get = requests.get(queue_url)
      queue_info_get.raise_for_status()
      queue_info = queue_info_get.json()
      queue_running = next(iter(queue_info.keys()))
      queue_running_value = queue_info[queue_running]
      
      if not queue_running_value:
        get_history_url = 'http://127.0.0.1:8188/api/history?max_items=64'
        history_dict = requests.get(get_history_url)
        history_dict.raise_for_status()
        history = history_dict.json()
        entry = history.get(str(prompt_id))
        if entry:
          outputs = entry.get("outputs",{})
          for output_data in outputs.values():
            images = output_data.get("images", [])
            for img in images:
              filename = img.get("filename")
              type = img.get("type")
            image_result_url = 'http://127.0.0.1:8188/api/view?filename='+filename+'&subfolder=&type='+type
            image_result = requests.get(image_result_url)
            image_result.raise_for_status()

            image_bytes = BytesIO(image_result.content)
            pil_image = Image.open(image_bytes)
            return pil_image
      
      time.sleep(3)

demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["image"],
)

demo.launch()