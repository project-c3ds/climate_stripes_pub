# Supporting codebase and social media data identifiers for "Bodies, banners and buses: How do people engage with the Climate Stripes".

This repository contains publishable dataset information and analysis code to derive the results presented in the paper "Bodies, banners and buses: How do people engage with the Climate Stripes". File information is given below. All scripts were tested using `Python 3.12.3` and the package versions specified in `requirements.txt`.

- `data` contains all publishable data extracts.
	- `tweet_ids.txt` contains the tweet IDs that comprise our dataset. Each line is a single ID.
- `gemma_identify_stripes.py` reads all images and passes them through the first step filter to find evidence of potential presence of the Climate Stripes. It uses a simple prompt for a binary classification via the `gemma-3-12b-it` instance hosted on HuggingFace.