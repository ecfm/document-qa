import pytest
import streamlit as st

@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit commands that would normally require a browser."""
    with st.container():
        yield 