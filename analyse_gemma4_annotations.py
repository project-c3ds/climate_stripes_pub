# This script loads and processes the LLM annotations and reports some summary stats.

import pandas as pd
import json
import code
import collections
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parameters.
annotation_data = 'all_stripes_annotations.jsonl'

# Load the annotations dataset.
annotation_data = pd.read_json(annotation_data,lines=True)

# Delete the rows with errors, they have been rerun and have another row in the data.
annotation_data = annotation_data[annotation_data['error'].isna()]

# Filter to only those images that contain the Stripes.
annotation_data = annotation_data[annotation_data['stripes_present']==1]

# Now apply some cleaning of terms to increas convergence.
term_maps = {'cross-stitch': 'art and hobbies',
			'drawing': 'art and hobbies',
			'embroidery': 'art and hobbies',
			'fabric': 'art and hobbies',
			'knitted fabric': 'art and hobbies',
			'painting': 'art and hobbies',
			'illustration': 'art and hobbies',
			'knitted piece': 'art and hobbies',
			'textile': 'art and hobbies',
			'crochet': 'art and hobbies',
			'woven fabric': 'art and hobbies',
			'crochet blanket': 'art and hobbies',
			'knitted blanket': 'art and hobbies',
			'knitted wall hanging': 'art and hobbies',
			'pie': 'art and hobbies',
			'artwork': 'art and hobbies',
			'bowl' : 'art and hobbies',
			'cake': 'art and hobbies',
			'cookie': 'art and hobbies',
			'glass tiles': 'art and hobbies',
			'heart-shaped cutout': 'art and hobbies',
			'lego model': 'art and hobbies',
			'sculpture': 'art and hobbies',
			'shipping container': 'art and hobbies',
			'string art': 'art and hobbies',
	
			 'virtual background': 'background',
			 'border': 'background', 
			 'backdrop': 'background',
			 'background screen': 'background',
	
			'button': 'badge',
			'pin': 'badge',
			'badge holder': 'badge',

			'tote bag': 'bag',

			'cloth': 'banner',

			'bicycle frame': 'bike',
			'cargo bike': 'bike',

			'digital billboard': 'billboard',

			'notebook': 'book cover',
			'book': 'book cover',

			'water bottle': 'bottle',

			'bus wrap': 'bus',

			'car wrap': 'car',
			'race car': 'car',

			'shawl': 'cape',

			'box': 'coaster',

			'dog bandana': 'dog coat',

			'mask': 'face mask',

			'floor mat': 'floor',
			'ground': 'floor',
			'rug': 'floor',

			'dolphin graphic': 'graphic',

			'earrings': 'jewellery',
			'necklace pendant': 'jewellery',
			'necklace': 'jewellery',
			'pendant': 'jewellery',
			'jewelry': 'jewellery',
			'earring': 'jewellery',
			'cufflink': 'jewellery',

			'trousers': 'legwear',
			'leggings': 'legwear',

			'light projection': 'light installation',

			'art print': 'magazine',  
			'magazine cover': 'magazine',
			
			'cup': 'mug',

			'building mural': 'mural',
			'wall mural': 'mural',
			'wall tiles': 'mural',

			'phone case': 'phone',
			'phone wallpaper': 'phone',
			'smartphone': 'phone',

			'piano keys': 'piano',
			
			'presentation slide': 'projection',

			'framed puzzle': 'puzzle',

			'scarves': 'scarf',

			'laptop screen': 'screen',
			'monitor': 'screen',
			'computer screen': 'screen',
			'tablet screen': 'screen',
			'digital display': 'screen',

			'shirt cuff': 'shirt',
			
			'socks': 'sock',

			'pavement': 'street art',
			'ground painting': 'street art',
			'road painting': 'street art',
			'bollard cover': 'street art',
			'bollard': 'street art',
			'chalk drawing': 'street art',

			'sweatshirt': 'sweater',
			
			'skin': 'tattoo',

			'truck wrap': 'truck',

			't-shirt sleeve': 't-shirt',
			'jersey': 't-shirt',  
			'cycling kit': 't-shirt', 
			'cycling jersey': 't-shirt',
			
			'fence': 'wall',

			'tapestry': 'wall hanging',  

			'website header': 'website',
			'website background': 'website',
			'social media banner': 'website'}

how_stripes_lists = annotation_data['how_stripes']  ## Pull out the lists
how_stripes_terms = [list(set([l.strip().lower() if l not in term_maps else term_maps[l.strip().lower()] for l in set(i)])) for i in how_stripes_lists]  ## Pull out the unique terms for each image.
how_stripes_terms = [i for l in how_stripes_terms for i in l]

how_stripes_counts = collections.Counter(how_stripes_terms)

# Output some summary stats. Total images at each level of filtering, number of labels applied.
print(f'Total number of images: {len(maybe_stripes)+len(not_stripes)}')
print(f'Number of images thought to be stripes: {len(maybe_stripes)}')
print(f'Number of images that had a stripe source identified: {len([k for k in valid_keys if annotation_data[k]['how_stripes']])}')

print(f'The 20 most common categories are:')
for i in range(20):
	print(how_stripes_counts.most_common(20)[i])

code.interact(local=locals())