# Data Acquisition Scripts

1. **Install dependencies:**
   ```bash
   uv pip install --group data-acquisition
   ```

2. **Download data:**
   ```bash
   TALKBANK_COOKIE="..." uv run python scripts/acquire_data.py
   ```

3. **Process data:**
   ```bash
   uv run python scripts/process_data.py
   ```