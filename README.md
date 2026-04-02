# Dataset Operations Tool

Local HTML tool for weekly dataset operations around rolling eval refresh, hard-case training append, and monthly benchmark refresh.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already.
2. Start the local web app:

```bash
uv run app.py
```

4. Open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## App Flow

1. Upload the QA dataset and class distribution or benchmark dataset on the home page.
2. Optionally upload historical rolling eval, benchmark eval, and training library.
3. Edit the config directly in the home page form instead of uploading a YAML file.
4. Use the top navigation to run:
   - weekly rolling eval update
   - training library update
   - monthly benchmark refresh
5. Download the generated CSVs and run log from the home page.

## Notes

- CSV and XLSX uploads are supported.
- Column names are configurable in `configs/config.yaml`.
- Dedup uses `item_id`.
- Sampling is seed-controlled for reproducibility.
- The benchmark eval page follows the original script logic: fixed start/end date plus P0/P1, medium, and longtail sample sizes derived from the distribution table.
- The current UI is server-rendered HTML using Flask and Jinja templates in `templates/`.
