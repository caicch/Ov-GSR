import os
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["FLASH_ATTENTION"] = "0"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
import argparse
from tqdm import tqdm

import torch

from clip.clip import load, tokenize

action_labels = []

with open('/root/autodl-tmp/data/SWiG_jsons/verb_indices.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    verb = line.split('\n')[0]
    action_labels.append(verb)

# def main(args):

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _ = load('ViT-B/32', device=device, jit=False)
model = model.eval()

# unseen_labels = open(os.path.join(data_path, "Concepts81.txt")).readlines()
# seen_labels = open(os.path.join(data_path, "Concepts925.txt")).readlines()

# label1006 = seen_labels + unseen_labels

label_token = torch.cat([tokenize(f"{c.strip()}") for c in action_labels]).to(device)
label_embed = torch.zeros((label_token.shape[0], 512))

with torch.no_grad():
    for i, label in enumerate(tqdm(label_token)):
        label_embed[i] = model.encode_text(label.unsqueeze(0))

torch.save(label_embed, os.path.join('/root/autodl-tmp/data/SWiG_jsons/', "label_Single_Action_b32.pt")) #1006, 768

print("Embedding Shape:", label_embed.shape)
print("Done!")


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-path", type=str, default='/mnt/data/NUS-WIDE/NUS-WIDE/')
#     #parser.add_argument("--clip-path", type=str, default=None)
#     args = parser.parse_args()
#
#     main(args)
