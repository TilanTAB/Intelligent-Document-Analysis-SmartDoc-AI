"""
Tests for the RelevanceChecker.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

import sys
sys.path.insert(0, '.')


class TestRelevanceChecker:
    """Test suite for RelevanceChecker."""
    
    @pytest.fixture
    def mock_parameters(self, monkeypatch):
        """Mock parameters to avoid API key requirement."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_key_for_testing")
    
    @pytest.fixture
    def context_validator(self, mock_parameters, fake_llm):
        """Create a RelevanceChecker with mocked LLM."""
        from intelligence.context_validator import RelevanceChecker
        checker = RelevanceChecker()
        checker.llm = fake_llm
        return checker
    
    def test_check_can_answer(self, context_validator, fake_retriever):
        """Test when documents can fully answer the question."""
        context_validator.llm.content = "CAN_ANSWER"
        
        result = context_validator.check(
            question="What is the PUE in Singapore?",
            retriever=fake_retriever,
            k=3
        )
        
        assert result == "CAN_ANSWER"
        assert fake_retriever.invoke_count == 1
    
    def test_check_partial_match(self, context_validator, fake_retriever):
        """Test when documents partially match the question."""
        context_validator.llm.content = "PARTIAL"
        
        result = context_validator.check(
            question="What is the historical trend of PUE?",
            retriever=fake_retriever,
            k=3
        )
        
        assert result == "PARTIAL"
    
    def test_check_no_match(self, context_validator, fake_retriever):
        """Test when documents don't match the question."""
        context_validator.llm.content = "NO_MATCH"
        
        result = context_validator.check(
            question="What is the weather in Paris?",
            retriever=fake_retriever,
            k=3
        )
        
        assert result == "NO_MATCH"
    
    def test_check_empty_question(self, context_validator, fake_retriever):
        """Test with empty question returns NO_MATCH."""
        result = context_validator.check(
            question="",
            retriever=fake_retriever,
            k=3
        )
        
        assert result == "NO_MATCH"
    
    def test_check_empty_retriever_results(self, context_validator, empty_retriever):
        """Test when retriever returns no documents."""
        result = context_validator.check(
            question="Any question",
            retriever=empty_retriever,
            k=3
        )
        
        assert result == "NO_MATCH"
    
    def test_check_invalid_llm_response(self, context_validator, fake_retriever):
        """Test when LLM returns invalid response."""
        context_validator.llm.content = "INVALID_LABEL"
        
        result = context_validator.check(
            question="What is the PUE?",
            retriever=fake_retriever,
            k=3
        )
        
        assert result == "NO_MATCH"
    
    def test_check_retriever_exception(self, context_validator):
        """Test when retriever throws an exception."""
        failing_retriever = MagicMock()
        failing_retriever.invoke.side_effect = Exception("Connection error")
        
        result = context_validator.check(
            question="Any question",
            retriever=failing_retriever,
            k=3
        )
        
        assert result == "NO_MATCH"
    
    def test_check_invalid_k_value(self, context_validator, fake_retriever):
        """Test with invalid k value defaults to 3."""
        context_validator.llm.content = "CAN_ANSWER"
        
        result = context_validator.check(
            question="What is the PUE?",
            retriever=fake_retriever,
            k=-1
        )
        
        assert result == "CAN_ANSWER"
