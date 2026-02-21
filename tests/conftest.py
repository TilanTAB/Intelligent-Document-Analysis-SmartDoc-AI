import os


# Ensure settings validation passes in test environments without real secrets.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("VISION_PROVIDER", "none")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-azure-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
