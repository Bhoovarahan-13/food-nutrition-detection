# Food Nutrition & Dietary Assistant (Google Colab)

This project is a research prototype for:
- Module 1: food image captioning using BLIP and food-name extraction.
- Module 2: nutrition estimation using `kb.csv` and serving multipliers.
- Web interface: Gradio app for uploading images and viewing food/nutrition results.

## Run in Google Colab

1. Upload `calorie_tracker_project_updated.zip` to Colab.
2. Run:

```python
!unzip -o calorie_tracker_project_updated.zip
%cd calorie_tracker
!pip install -r requirements.txt
```

3. Launch the web app:

```python
!python app_gradio.py
```

4. Wait for the Gradio public URL (`https://....gradio.live`) and open it.
   Keep the Colab runtime running while using the link.

The first automatic image analysis downloads the BLIP model and may take several minutes.
A Hugging Face token is generally optional for public models, but unauthenticated downloads
may have lower rate limits.

## Using the app

- Upload a JPG/PNG meal photo.
- Optionally describe the meal.
- To correct missed foods, enter food names separated by semicolons in **Food corrections**.
  If supplied, this list overrides automatic BLIP detection.
- Optionally set portion multipliers, e.g. `white rice=1.5; chicken breast=1`.
  Multipliers scale the per-serving nutrition values in `kb.csv`.
- Click **Analyse meal**.

## Important limitations

BLIP is a general image-captioning model, not a dedicated food detector. It can miss ingredients
or hallucinate food items. Nutrition totals are knowledge-base estimates, not measurements from
the image. Serving sizes must be confirmed manually. The current prototype does not yet provide
personalized dietary recommendations based on a user's health profile.

For evaluation, use correctly labelled images and real meal nutrition totals. Do not present
the sample results as clinical or validated nutrition estimates.
