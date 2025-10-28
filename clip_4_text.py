import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["FLASH_ATTENTION"] = "0"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
import sys
sys.modules["flash_attn"] = None
sys.modules["flash_attn_2_cuda"] = None

from tqdm import tqdm
from PIL import Image
import json
import torch
from transformers import AutoTokenizer, CLIPTextModel

# Set CUDA environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# File paths
ann_file = '/root/autodl-tmp/data/SWiG_jsons/train_L10_with_obj_cap_refined.json'
out_path = '/root/autodl-tmp/text_clip_lage_16/'

# Load annotations
with open(ann_file) as file:
    SWiG_json = json.load(file)

# Process data
for name, value in tqdm(SWiG_json.items(), total=len(SWiG_json), desc="Processing"):
    full_image_name = name.split('.')[0]

    # Prepare captions
    captions = [
        SWiG_json[name]['pos_verb_caption'],
        SWiG_json[name]['neg_verb_caption'],
        SWiG_json[name]['pos_attr_caption'],
        SWiG_json[name]['neg_attr_caption']
    ]

    # Tokenize captions and move tensors to GPU
    caption_input = tokenizer(
        captions,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=77
    ).to(device)  # Move tokenized inputs to GPU

    # Perform inference on GPU
    with torch.no_grad():
        text_outputs = model(**caption_input)

    # Save output to GPU-compatible format
    torch.save(
        text_outputs.last_hidden_state.cpu(),  # Move tensor back to CPU before saving
        os.path.join(out_path, full_image_name + '.pt')
    )
