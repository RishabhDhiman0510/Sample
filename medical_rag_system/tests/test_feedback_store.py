"""Tests for FeedbackStore."""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from medical_rag.feedback.feedback_store import PIIRedactor

def test_pii_redactor():
    text = "Contact Dr. John Smith at john.smith@example.com or call 555-123-4567"
    redacted = PIIRedactor.redact(text)
    assert "<REDACTED_EMAIL>" in redacted or "john.smith" not in redacted
