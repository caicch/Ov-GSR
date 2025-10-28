import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
from transformers import Blip2Processor, Blip2Model, AutoProcessor, CLIPVisionModel
from PIL import Image
import torch, json
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load processor + model
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device) #laion/CLIP-ViT-bigG-14-laion2B-39B-b160k #openai/clip-vit-large-patch14-336
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Point to just the vision tower
vision_model = model.vision_model

# Load data
ann_file = "/root/autodl-tmp/data/SWiG_jsons/dev_ind.json"
out_path = "/root/autodl-tmp/data/image_512_clip_L_336"
os.makedirs(out_path, exist_ok=True)

with open(ann_file) as f:
    SWiG_json = json.load(f)

for name, _ in tqdm(SWiG_json.items(), total=len(SWiG_json), desc="Processing"):
    path = f"/root/autodl-tmp/data/images_512/{name}"
    im = Image.open(path).convert("RGB")

    # Processor outputs multiple things, but we only need pixel_values
    inputs = processor(images=im, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        vision_outputs = vision_model(pixel_values)
        features = vision_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

    torch.save(features, os.path.join(out_path, name.split('.')[0] + ".pt"))
    print(name, features.shape)
