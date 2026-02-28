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

4. **Label utterances with LLM:**
   
   Set up your `.env` file with:
   ```
   API_KEY=your_api_key
   BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   ```
   
   Then run the batch labeling process:
   
   **Step 1 - Create batch file:**
   ```bash
   uv run python scripts/label_data.py create
   ```
   This creates a batch file to input to an LLM.
   
   **Step 2 - Submit:**
   ```bash
   uv run python scripts/label_data.py batch_submit <Path to batchfile (from Step 1)>
   ```
   This submits (in chunks), downloads the results and updates `utterances.parquet` with error pattern labels.