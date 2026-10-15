# Supporting codebase and social media data identifiers for "Bodies, banners and buses: How do people engage with the Climate Stripes".

This repository contains publishable dataset information and analysis code to derive the results presented in the paper "Bodies, banners and buses: How do people engage with the Climate Stripes". File information is given below. All scripts were tested using `Python 3.12.3` and the package versions specified in `requirements.txt`.

- `data` contains all publishable data extracts.
	- `all_stripes_annotations.jsonl` contains the image annotations applied by the VLM to the social media images.
	- `all_stripes_annotations_continents.jsonl` contains the same image annotations, but with detected locations aggregated at the continent or ocean level.
	- `tweet_ids.txt` contains the tweet IDs that comprise our dataset. Each line is a single ID.

- `.env.example` is an example .env file illustrating the fields expected to be included.

- `annotate_images.py` details the image annotation process. It includes details for both the annotation prompt and the VLM used.

- `gemma_identify_stripes.py` reads all images and passes them through the first step filter to find evidence of potential presence of the Climate Stripes. It uses a simple prompt for a binary classification via the `gemma-3-12b-it` instance hosted on HuggingFace.

- `survey_analysis.py` reads the survey data and runs the statistical analyses on the responses and topic modelling on the survey responses.