import torch
from transformers import pipeline
from PIL import Image
import os
from huggingface_hub import login
import tqdm
import json

read_token = 'YOUR HF READ TOKEN HERE'  # Update this to the relevant read access token for your HF account. This is required to download the Gemma model.
login(token=read_token)


pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)

image_files = sorted([f for f in os.listdir('all_imgs') if f.endswith('.jpg')])

# Load past labelling attempts if interrupted.
try:
    with open('all_stripes_labels_gemma_12b.json','r') as f:
        labels = json.load(f)
except FileNotFoundError:
    labels = {}

for i,file in tqdm.tqdm(enumerate(image_files)):
    if labels.get(file) == None:
        
        try:
             
            image = Image.open('all_imgs/'+file)
        except:
            continue
        
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "The Climate Stripes are a visualisation of changing average temperature.These ‘warming stripe’ graphics are visual representations of the change in temperature as measured in each country, region or city over the past 100+ years. Each stripe or bar represents the temperature in that country, region or city averaged over a year. The stripes typically start around the year 1900 and finish in 2025, but for many countries, regions and cities the stripes start in the 19th century or sometimes even the 18th century. Some people use the Climate Stripes to decorate clothing, buildings and signage with its blue to red transition. Do the Climate Stripes feature in this image in any orientation? Reply 1 for yes, and 0 for no ONLY."}
            ]
            }
        ]

        output = pipe(text=messages, max_new_tokens=200)
        labels[file] = output[0]['generated_text'][1]['content']

        if i % 100 == 0:
            with open('all_stripes_labels_gemma_12b.json','w') as f:
	            f.write(json.dumps(labels))