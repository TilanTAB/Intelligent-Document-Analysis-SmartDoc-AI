"""
Tests for the VerificationAgent.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Import after setting up mocks to avoid API key validation
import sys
sys.path.insert(0, '.')


class TestVerificationAgent:
    """Test suite for VerificationAgent."""
    
    @pytest.fixture
    def mock_parameters(self, monkeypatch):
        """Mock parameters to avoid API key requirement."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_key_for_testing")
    
    @pytest.fixture
    def accuracy_verifier(self, mock_parameters, fake_llm):
        """Create a VerificationAgent with mocked LLM."""
        from intelligence.accuracy_verifier import VerificationAgent
        return VerificationAgent(llm=fake_llm)
    
    def test_check_with_supported_answer(self, accuracy_verifier, sample_documents):
        """Test verification with an answer supported by documents."""
        # Configure the fake LLM to return a supported response
        accuracy_verifier.llm.content = """
        Supported: YES
        Unsupported Claims: []
        Contradictions: []
        Relevant: YES
        Additional Details: The answer is well-supported by the context.
        """
        
        result = accuracy_verifier.check(
            answer="The PUE in Singapore was 1.12 in 2022.",
            documents=sample_documents
        )
        
        assert "verification_report" in result
        assert "Supported: YES" in result["verification_report"]
        assert "context_used" in result
    
    def test_check_with_unsupported_answer(self, accuracy_verifier, sample_documents):
        """Test verification with an unsupported answer."""
        accuracy_verifier.llm.content = """
        Supported: NO
        Unsupported Claims: [The PUE was 1.5]
        Contradictions: []
        Relevant: YES
        Additional Details: The claimed PUE value is not in the context.
        """
        
        result = accuracy_verifier.check(
            answer="The PUE in Singapore was 1.5 in 2022.",
            documents=sample_documents
        )
        
        assert "Supported: NO" in result["verification_report"]
    
    def test_parse_verification_response_valid(self, accuracy_verifier):
        """Test parsing a valid verification response."""
        response = """
        Supported: YES
        Unsupported Claims: []
        Contradictions: []
        Relevant: YES
        Additional Details: All claims verified.
        """
        
        parsed = accuracy_verifier.parse_verification_response(response)
        
        assert parsed["Supported"] == "YES"
        assert parsed["Relevant"] == "YES"
        assert parsed["Unsupported Claims"] == []
    
    def test_parse_verification_response_with_claims(self, accuracy_verifier):
        """Test parsing response with unsupported claims."""
        response = """
        Supported: NO
        Unsupported Claims: [claim1, claim2]
        Contradictions: [contradiction1]
        Relevant: YES
        Additional Details: Multiple issues found.
        """
        
        parsed = accuracy_verifier.parse_verification_response(response)
        
        assert parsed["Supported"] == "NO"
        assert len(parsed["Unsupported Claims"]) == 2
        assert len(parsed["Contradictions"]) == 1
    
    def test_format_verification_report(self, accuracy_verifier):
        """Test formatting a verification report."""
        verification = {
            "Supported": "YES",
            "Unsupported Claims": [],
            "Contradictions": [],
            "Relevant": "YES",
            "Additional Details": "Well verified."
        }
        
        report = accuracy_verifier.format_verification_report(verification)
        
        assert "**Supported:** YES" in report
        assert "**Relevant:** YES" in report
        assert "**Unsupported Claims:** None" in report
