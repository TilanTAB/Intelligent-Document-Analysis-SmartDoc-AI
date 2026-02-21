# Viewing Logs on Hugging Face Spaces

## Overview

Your SmartDoc AI application now logs user interactions (IP addresses, questions, file uploads) to help you understand usage patterns and improve the system.

---

## ?? What Gets Logged

Each user interaction is logged with:
- **Timestamp**: When the question was asked
- **IP Address**: User's IP (for analytics, not identification)
- **Question**: The actual question text
- **Question Length**: Character count
- **Number of Files**: How many documents uploaded
- **File Types**: Extensions (.pdf, .docx, etc.)
- **Success**: Whether the request completed successfully
- **Error**: Error message if the request failed

---

## ?? How to View Logs on Hugging Face Spaces

### Method 1: Real-Time Container Logs (Recommended)

1. **Go to your Space page:**
   ```
   https://huggingface.co/spaces/TilanB/smartdoc-ai
   ```

2. **Click on the "Logs" tab** at the top of the page

3. **View real-time logs:**
   - You'll see all application logs including user interactions
   - Logs are streamed in real-time as users interact with your app
   - Look for lines containing `[USER_ANALYTICS]` for interaction logs

4. **Filter logs:**
   - Use your browser's search (Ctrl+F / Cmd+F) to find specific IPs or questions
   - Example: Search for an IP like `"ip": "12.34.56.78"`

### Method 2: Download Log Files

Since HF Spaces use ephemeral storage, logs are stored in `/tmp/logs/` and will be lost on Space restart. To persist logs:

**Option A: Add a download button to your UI**

Add this code to create a download button for logs:

```python
# In your Gradio interface (main.py):
def download_logs():
    """Export logs to JSON file."""
    try:
        export_path = analytics_logger.export_logs()
        return export_path
    except Exception as e:
        logger.error(f"Failed to export logs: {e}")
        return None

# Add to your Gradio interface:
with gr.Tab("Admin"):
    download_btn = gr.Button("Download User Logs")
    log_file = gr.File(label="Log Export")
    
    download_btn.click(
        fn=download_logs,
        inputs=[],
        outputs=[log_file]
    )
```

**Option B: Use Persistent Dataset Storage**

Save logs to a Hugging Face dataset for persistence:

```python
# Add to your code:
from datasets import Dataset
import pandas as pd

def sync_logs_to_dataset():
    """Upload logs to HF dataset for persistence."""
    if not os.path.exists(analytics_logger.log_file):
        return
    
    # Read logs
    logs = []
    with open(analytics_logger.log_file, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    # Convert to dataset
    df = pd.DataFrame(logs)
    dataset = Dataset.from_pandas(df)
    
    # Push to HF (requires HF_TOKEN)
    dataset.push_to_hub(
        "TilanB/smartdoc-logs",
        token=os.environ.get("HF_TOKEN"),
        private=True  # Keep logs private!
    )
```

---

## ?? Usage Statistics

Your logger includes a `get_stats()` method to get quick analytics:

```python
stats = analytics_logger.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Unique IPs: {stats['unique_ips']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
```

Add this to your Gradio UI:

```python
def show_stats():
    """Display usage statistics."""
    stats = analytics_logger.get_stats()
    return f"""
### ?? Usage Statistics

- **Total Queries:** {stats['total_queries']}
- **Unique Users (IPs):** {stats['unique_ips']}
- **Success Rate:** {stats['success_rate']:.1f}%
"""

# Add to your Gradio interface:
with gr.Tab("Stats"):
    stats_display = gr.Markdown()
    refresh_stats_btn = gr.Button("Refresh Stats")
    
    refresh_stats_btn.click(
        fn=show_stats,
        inputs=[],
        outputs=[stats_display]
    )
```

---

## ?? Privacy & Security Best Practices

### 1. **IP Address Handling**

IPs are logged for analytics, not identification:
- Use for abuse detection (rate limiting)
- Analyze geographic patterns (if needed)
- **DO NOT** attempt to identify specific users

### 2. **Question Content**

Questions may contain sensitive information:
- **DO NOT** share logs publicly
- Store logs securely (private HF dataset or encrypted storage)
- Consider anonymizing logs after analysis

### 3. **GDPR Compliance**

If you have EU users, consider:
- Adding a privacy notice to your app
- Providing a way for users to request data deletion
- Only keeping logs for necessary duration (e.g., 30 days)

Example privacy notice:

```python
gr.Markdown("""
### ?? Privacy Notice
We log anonymized usage data (questions, timestamps) for analytics and improvement purposes. 
No personal information is stored. Logs are retained for 30 days.
""")
```

---

## ?? Log File Format

Logs are stored in **JSON Lines** format (`.jsonl`):

```json
{"timestamp": "2026-01-01T12:34:56.789", "ip": "12.34.56.78", "question": "What is AI?", "question_length": 11, "num_files": 1, "file_types": [".pdf"], "success": true, "error": null}
{"timestamp": "2026-01-01T12:35:12.345", "ip": "98.76.54.32", "question": "Explain charts", "question_length": 14, "num_files": 2, "file_types": [".pdf", ".docx"], "success": true, "error": null}
```

You can analyze this with:
- **Python:** `json.loads(line)` for each line
- **pandas:** `pd.read_json(log_file, lines=True)`
- **jq:** Command-line JSON processor

---

## ??? Analyzing Logs

### Python Script to Analyze Logs:

```python
import json
from collections import Counter
from datetime import datetime

def analyze_logs(log_file="logs/user_interactions.jsonl"):
    """Analyze user interaction logs."""
    
    ips = []
    questions = []
    file_types = []
    errors = []
    timestamps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            
            ips.append(entry['ip'])
            questions.append(entry['question'])
            file_types.extend(entry.get('file_types', []))
            timestamps.append(entry['timestamp'])
            
            if not entry.get('success', True):
                errors.append(entry.get('error', 'Unknown'))
    
    print("=== Usage Analytics ===")
    print(f"\nTotal Queries: {len(questions)}")
    print(f"Unique IPs: {len(set(ips))}")
    print(f"\nTop 5 IPs:")
    for ip, count in Counter(ips).most_common(5):
        print(f"  {ip}: {count} queries")
    
    print(f"\nMost Common File Types:")
    for ftype, count in Counter(file_types).most_common(5):
        print(f"  {ftype}: {count} files")
    
    if errors:
        print(f"\nErrors ({len(errors)} total):")
        for error, count in Counter(errors).most_common(5):
            print(f"  {error}: {count} times")
    
    print(f"\nAverage Question Length: {sum(len(q) for q in questions) / len(questions):.1f} chars")

if __name__ == "__main__":
    analyze_logs()
```

---

## ?? Monitoring & Alerts

For production, consider setting up alerts:

1. **High error rate:** Alert if >10% of requests fail
2. **Unusual traffic:** Alert if requests spike >5x normal
3. **Abuse detection:** Alert if single IP exceeds rate limit repeatedly

---

## ?? Example Queries

### Find all questions from a specific IP:
```bash
grep '"ip": "12.34.56.78"' logs/user_interactions.jsonl
```

### Count total successful queries:
```bash
grep '"success": true' logs/user_interactions.jsonl | wc -l
```

### Extract all unique questions:
```bash
jq -r '.question' logs/user_interactions.jsonl | sort -u
```

### Find all failed requests:
```bash
jq 'select(.success == false)' logs/user_interactions.jsonl
```

---

## ? Summary

- **Real-time logs:** View in HF Spaces "Logs" tab
- **Persistent storage:** Use HF datasets or export button
- **Analytics:** Use `get_stats()` or analyze `.jsonl` file
- **Privacy:** Keep logs private, anonymize if needed
- **Monitoring:** Set up alerts for errors and abuse

Your logs are now helping you understand how users interact with SmartDoc AI! ??
