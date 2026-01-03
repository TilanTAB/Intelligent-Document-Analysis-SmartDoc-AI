# Security Recommendations - SmartDoc AI

**Priority:** 🔴 HIGH  
**Review Date:** 2026-01-02  
**Status:** Action Required

---

## 🚨 Critical Security Vulnerabilities

### 1. Unsafe Pickle Deserialization

**Severity:** 🔴 CRITICAL  
**CWE:** CWE-502 (Deserialization of Untrusted Data)  
**Location:** `content_analyzer/document_parser.py:251-267`

**Vulnerable Code:**
```python
def _load_from_cache(self, cache_path: Path) -> List:
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)  # ⚠️ UNSAFE!
```

**Attack Scenario:**
1. Attacker gains write access to cache directory
2. Places malicious pickle file in cache
3. Application deserializes and executes arbitrary code

**Fix (Option 1 - JSON):**
```python
import json
from typing import List, Dict

def _save_to_cache(self, chunks: List, cache_path: Path):
    """Save chunks to JSON format."""
    data = {
        "timestamp": datetime.now().timestamp(),
        "chunks": [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)

def _load_from_cache(self, cache_path: Path) -> List[Document]:
    """Load chunks from JSON format."""
    with open(cache_path, "r") as f:
        data = json.load(f)
    
    chunks = [
        Document(
            page_content=c["page_content"],
            metadata=c["metadata"]
        )
        for c in data["chunks"]
    ]
    return chunks
```

**Fix (Option 2 - Signed Pickle):**
```python
import hmac
import hashlib
from pathlib import Path

SECRET_KEY = os.environ.get("CACHE_SECRET_KEY")  # Load from env

def _save_to_cache(self, chunks: List, cache_path: Path):
    data = {
        "timestamp": datetime.now().timestamp(),
        "chunks": chunks
    }
    serialized = pickle.dumps(data)
    signature = hmac.new(SECRET_KEY.encode(), serialized, hashlib.sha256).hexdigest()
    
    with open(cache_path, "wb") as f:
        f.write(signature.encode() + b"\n" + serialized)

def _load_from_cache(self, cache_path: Path) -> List:
    with open(cache_path, "rb") as f:
        content = f.read()
    
    signature, serialized = content.split(b"\n", 1)
    expected_sig = hmac.new(SECRET_KEY.encode(), serialized, hashlib.sha256).hexdigest()
    
    if not hmac.compare_digest(signature.decode(), expected_sig):
        raise ValueError("Cache file signature invalid - possible tampering")
    
    return pickle.loads(serialized)
```

---

### 2. Path Traversal Vulnerability

**Severity:** 🔴 CRITICAL  
**CWE:** CWE-22 (Path Traversal)  
**Location:** `content_analyzer/document_parser.py:281-285`

**Vulnerable Code:**
```python
def _process_file(self, file) -> List[Document]:
    file_ext = Path(file.name).suffix.lower()  # No validation
```

**Attack Scenario:**
```python
# Attacker uploads file with path: ../../../../etc/passwd
# Or: /var/www/sensitive_data.txt
```

**Fix:**
```python
import os
from pathlib import Path

UPLOAD_DIR = Path("/tmp/uploads").resolve()

def _process_file(self, file) -> List[Document]:
    """Process file with path traversal protection."""
    # Resolve to absolute path
    file_path = Path(file.name).resolve()
    
    # Ensure file is within allowed directory
    try:
        file_path.relative_to(UPLOAD_DIR)
    except ValueError:
        raise ValueError(f"Invalid file path: {file.name}")
    
    # Additional checks
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    file_ext = file_path.suffix.lower()
    # ... continue processing
```

---

### 3. Missing Content-Type Validation

**Severity:** 🟠 HIGH  
**CWE:** CWE-434 (Unrestricted Upload of File with Dangerous Type)  
**Location:** `content_analyzer/document_parser.py:210-216`

**Vulnerable Code:**
```python
# Only checks file extension
file_ext = Path(file.name).suffix.lower()
if file_ext not in ALLOWED_TYPES:
    raise ValueError(f"File type {file_ext} not supported")
```

**Attack Scenario:**
1. Attacker renames malicious executable: `malware.exe` → `malware.pdf`
2. Extension check passes
3. File processed, potentially executing malicious code

**Fix:**
```python
import magic  # python-magic library

ALLOWED_MIME_TYPES = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "text/plain": [".txt"],
    "text/markdown": [".md"],
}

def validate_files(self, files: List) -> bool:
    for file in files:
        # Check extension
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in ALLOWED_TYPES:
            raise ValueError(f"Invalid extension: {file_ext}")
        
        # Verify actual content type (magic bytes)
        try:
            mime_type = magic.from_file(file.name, mime=True)
        except Exception as e:
            raise ValueError(f"Cannot determine file type: {e}")
        
        # Ensure MIME type matches extension
        if mime_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported file type: {mime_type}")
        
        if file_ext not in ALLOWED_MIME_TYPES[mime_type]:
            raise ValueError(
                f"File extension {file_ext} does not match content type {mime_type}"
            )
    
    return True
```

**Installation:**
```bash
pip install python-magic
# On Ubuntu/Debian:
sudo apt-get install libmagic1
```

---

### 4. Rate Limiting Bypass

**Severity:** 🟠 HIGH  
**CWE:** CWE-770 (Allocation of Resources Without Limits)  
**Location:** `main.py:24-45`

**Vulnerable Code:**
```python
# In-memory storage (lost on restart)
_calls = defaultdict(deque)

def rate_limit(request):
    ip = getattr(request.client, "host", "unknown")  # Easy to spoof
    # No X-Forwarded-For checking
    # No persistent storage
```

**Attack Scenario:**
1. Attacker uses VPN/proxy to change IP
2. Restarting service resets all rate limits
3. Distributed attack from multiple IPs

**Fix (Redis-based):**
```python
import redis
import time
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

WINDOW_S = 3600
MAX_CALLS = 5

def rate_limit(request):
    """Enhanced rate limiting with Redis backend."""
    # Get real IP (considering proxies)
    ip = request.headers.get("X-Forwarded-For", 
                             request.headers.get("X-Real-IP",
                             request.client.host))
    
    # Use first IP if multiple
    if "," in ip:
        ip = ip.split(",")[0].strip()
    
    # Get user session if available (better than IP)
    session_id = request.session.get("session_id")
    rate_key = f"rate_limit:{session_id or ip}"
    
    now = int(time.time())
    window_start = now - WINDOW_S
    
    # Remove old entries
    redis_client.zremrangebyscore(rate_key, 0, window_start)
    
    # Count requests in current window
    request_count = redis_client.zcard(rate_key)
    
    if request_count >= MAX_CALLS:
        # Get time until next allowed request
        oldest_timestamp = redis_client.zrange(rate_key, 0, 0, withscores=True)
        if oldest_timestamp:
            wait_time = int(oldest_timestamp[0][1] + WINDOW_S - now)
            raise gr.Error(
                f"Rate limit exceeded. Please wait {wait_time} seconds."
            )
        raise gr.Error("Rate limit exceeded. Please try again later.")
    
    # Add current request
    redis_client.zadd(rate_key, {str(now): now})
    redis_client.expire(rate_key, WINDOW_S)
    
    return True
```

---

### 5. API Key Exposure

**Severity:** 🟠 HIGH  
**CWE:** CWE-798 (Use of Hard-coded Credentials)  
**Location:** `configuration/parameters.py:42-45`

**Vulnerable Code:**
```python
GOOGLE_API_KEY: str = Field(...)  # Stored in plain text .env file
```

**Risks:**
- Keys in environment variables
- Keys in logs/error messages
- Keys in Git history (if .env committed)

**Fix (AWS Secrets Manager):**
```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str, region_name: str = "us-east-1") -> str:
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        logger.error(f"Failed to retrieve secret: {e}")
        raise

class Settings(BaseSettings):
    # Override default behavior
    @property
    def GOOGLE_API_KEY(self) -> str:
        if os.environ.get("ENVIRONMENT") == "production":
            return get_secret("smartdoc-ai/google-api-key")
        return os.environ.get("GOOGLE_API_KEY", "")
```

**Fix (Azure Key Vault):**
```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def get_secret_azure(secret_name: str, vault_url: str) -> str:
    """Retrieve secret from Azure Key Vault."""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    secret = client.get_secret(secret_name)
    return secret.value
```

---

## Additional Security Recommendations

### 6. Input Sanitization

**Add to all user inputs:**
```python
import html
import re

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Limit length
    text = text[:max_length]
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # HTML escape
    text = html.escape(text)
    
    return text.strip()

# Use in process_question:
def process_question(question_text, ...):
    question_text = sanitize_input(question_text, max_length=500)
    # ... continue
```

### 7. Secure Logging

**Prevent PII exposure:**
```python
import re

def sanitize_log_message(message: str) -> str:
    """Remove sensitive data from log messages."""
    # Remove email addresses
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '[EMAIL]', message)
    
    # Remove API keys (pattern: starts with letters, 32+ chars)
    message = re.sub(r'\b[A-Za-z]{2,}[A-Za-z0-9_-]{30,}\b', '[API_KEY]', message)
    
    # Remove phone numbers
    message = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', message)
    
    return message

# Custom log handler
class SecureLogHandler(logging.Handler):
    def emit(self, record):
        record.msg = sanitize_log_message(str(record.msg))
        # ... continue with normal logging
```

### 8. Dependency Security

**Add to CI/CD:**
```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Safety Check
        run: |
          pip install safety
          safety check --json
      
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
```

---

## Security Checklist

- [ ] Replace pickle with JSON for cache
- [ ] Add path traversal protection
- [ ] Implement content-type validation
- [ ] Use persistent rate limiting (Redis)
- [ ] Migrate to secrets manager
- [ ] Add input sanitization
- [ ] Implement secure logging
- [ ] Set up dependency scanning
- [ ] Add security headers to Gradio
- [ ] Enable HTTPS in production
- [ ] Implement audit logging
- [ ] Add CSRF protection
- [ ] Regular security audits
- [ ] Penetration testing

---

## Deployment Security

### Production Checklist
```bash
# 1. Environment setup
export ENVIRONMENT=production
export LOG_LEVEL=WARNING  # Reduce verbose logging

# 2. File permissions
chmod 700 document_cache/
chmod 600 .env

# 3. Firewall rules
sudo ufw allow 7860/tcp  # Only Gradio port
sudo ufw enable

# 4. Process isolation
# Run as non-root user
useradd -r -s /bin/false smartdoc
sudo -u smartdoc python main.py

# 5. Resource limits
ulimit -n 1024  # Max open files
ulimit -u 50    # Max processes
```

---

## Contact & Incident Response

**Security Issues:** Report immediately to security@example.com  
**Bug Bounty:** Consider setting up program for external researchers  
**Incident Response Plan:** Document in separate SECURITY.md file

---

**Last Updated:** 2026-01-02  
**Next Review:** 2026-02-02 (monthly reviews recommended)
