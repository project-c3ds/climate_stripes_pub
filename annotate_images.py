"""Annotate climate-stripes images with a VLM on Verda (DataCrunch) serverless.

Config comes from .env: CHAT_MODEL, API_KEY, API_BASE_URL.
Needs: pip install openai pydantic python-dotenv tqdm

Usage:
    python annotate_images.py        # annotate everything not yet done
    python annotate_images.py 5      # annotate only the next 5, for a quick check

Results are appended to a jsonl file, one line per image, so a run can be
interrupted and restarted without losing work.
"""

import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

IMG_DIR = "all_imgs"
OUT = "all_stripes_annotations.jsonl"
WORKERS = 16  # throughput saturates around here; 32 and 64 measured no faster

# The description of the Climate Stripes is kept word for word from the prompt in
# gemma_annotate_images.py; the rules under "Rules:" were added after scoring against
# the human labels and need the domain expert's sign-off.
SYSTEM_PROMPT = """\
You are an expert image annotator for a research project on the Climate Stripes.

The Climate Stripes are a visualisation of changing average temperature. These 'warming
stripe' graphics are visual representations of the change in temperature as measured in
each country, region or city over the past 100+ years. Each stripe or bar represents the
temperature in that country, region or city averaged over a year. The stripes typically
start around the year 1900 and finish in 2025, but for many countries, regions and cities
the stripes start in the 19th century or sometimes even the 18th century. Some people use
the Climate Stripes to decorate clothing, buildings and signage with its blue to red
transition.

How do the Climate Stripes feature in this image (in any orientation), and does the image
indicate the region it represents? They could appear on a poster, an item of clothing,
projected on a building, or the image could be the Climate Stripes themselves. Some images
may not contain locations, or the Climate Stripes at all. In such cases, respond with empty
lists as appropriate.

For each image you are given, report:
- stripes_present: true if the Climate Stripes appear anywhere in the image, false if not.
- how_stripes: every distinct way the Climate Stripes appear in the image
  (e.g. "figure", "poster", "scarf", "t-shirt", "mug", "building projection", "flag",
  "banner", "tattoo"). Use short lowercase noun phrases. Empty list if the Climate
  Stripes are not present.
- locations: every place the stripes are labelled as representing, read from text or
  unambiguous context in the image (e.g. "London", "Germany"). Empty list if none.

Rules:
- Only report what is visible. Never guess a location from the setting or scenery.
- A blue-to-red gradient is not enough on its own; the stripes are discrete bars. The bars
  may all be the same height, or vary in height with the temperature anomaly above and
  below a baseline. Both are the Climate Stripes.
- Report the object the stripes are on, not the scene around it.
- The Climate Stripes show temperature. A chart of some other variable (drought index,
  rainfall, sea ice) is not the Climate Stripes even when drawn as blue and red bars.
- The Climate Stripes use only blues, near-white and reds, and run from cooler colours at
  one end to warmer at the other. A striped design in other colours (yellow, green, pink),
  or with no cool-to-warm progression, is not the Climate Stripes even though it is
  striped and contains red and blue. Sports-team stripes and charity branding are common
  examples.
- Decorative and everyday use does count. Stripes on clothing, a bag, a mug, a building or
  a mural are the Climate Stripes if the pattern is right, even with no title, axis,
  year labels or legend.
- A national symbol shown with the stripes (flag, landmark, country outline) counts as a
  location. Report the country, not the city the landmark stands in.
- stripes_present is judged on its own. If the stripes are there but you cannot name the
  object they are on, set it true and leave how_stripes empty.
"""


class Annotation(BaseModel):
    """Field descriptions document the model for us; the semantics the model
    actually follows live in SYSTEM_PROMPT, since guided decoding only
    constrains the token grammar."""

    stripes_present: bool = Field(description="Whether the Climate Stripes appear in the image at all")
    how_stripes: list[str] = Field(description="Ways the Climate Stripes appear in the image")
    locations: list[str] = Field(description="Locations the Climate Stripes are labelled as representing")


load_dotenv()

MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"

# max_retries covers rate limits, 5xx and serverless cold starts.
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"],
    max_retries=5,
    timeout=120,
)

SCHEMA = Annotation.model_json_schema() | {"additionalProperties": False}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "annotation", "schema": SCHEMA, "strict": True},
}


def data_url(path):
    """Read an image and encode it for the image_url message field."""
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def ask_model(file):
    """Send one image to the model and return its validated annotation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(os.path.join(IMG_DIR, file))}},
                {"type": "text", "text": "Annotate this image."},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format=RESPONSE_FORMAT,
        max_tokens=800,  # a few images list enough locations to truncate at 300
        temperature=0,
    )
    annotation = Annotation.model_validate_json(response.choices[0].message.content)
    return annotation.model_dump()


def annotate(file):
    """Annotate one image, recording the error instead of raising if it fails."""
    try:
        return {"file": file, "model": MODEL, **ask_model(file)}
    except Exception as error:
        return {"file": file, "model": MODEL, "error": str(error)}


def load(path=OUT):
    """Read the results file into a dict keyed by filename.

    A retried image has more than one line; the last one wins.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        results = [json.loads(line) for line in f]
    return {result["file"]: result for result in results}


def images_to_do(limit=None):
    """List the images still needing annotation, newest failures included."""
    results = load()
    done = {file for file, result in results.items() if "error" not in result}

    todo = []
    empty = 0
    for file in sorted(os.listdir(IMG_DIR)):
        if not file.endswith(".jpg") or file in done:
            continue
        if os.path.getsize(os.path.join(IMG_DIR, file)) == 0:
            empty += 1  # download failed at collection time, nothing to send
            continue
        todo.append(file)

    if empty:
        print(f"Skipping {empty} empty images (failed downloads)")
    return todo[:limit] if limit else todo


def main(limit=None):
    todo = images_to_do(limit)
    already_done = len(load())
    print(f"Annotating {len(todo)} images with {MODEL} using {WORKERS} workers")
    if already_done:
        print(f"Resuming, {already_done} already in {OUT}")

    stripes = errors = 0
    bar = tqdm.tqdm(total=len(todo), unit="img", smoothing=0.05)

    with open(OUT, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
        for result in pool.map(annotate, todo):
            out.write(json.dumps(result) + "\n")
            out.flush()

            if "error" in result:
                errors += 1
            elif result["stripes_present"]:
                stripes += 1
            bar.set_postfix(stripes=stripes, errors=errors, refresh=False)
            bar.update()

    bar.close()
    print(f"Done. {stripes} images with stripes, {errors} errors, written to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
