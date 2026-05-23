"""Tests for the API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.responses import AnalyzeResponse


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_analyze_response() -> AnalyzeResponse:
    """Return a sample AnalyzeResponse for mocking."""
    return AnalyzeResponse(
        selected_model="phi3:mini",
        routing_decision="slm",
        confidence_score=0.85,
        latency_ms=120.5,
        final_response="Revenue increased by 15% year-over-year.",
    )


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 with healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestAnalyzeEndpoint:
    """Tests for the POST /analyze endpoint."""

    @patch("app.api.routes.analyze", new_callable=AsyncMock)
    def test_analyze_valid_input_returns_200(
        self,
        mock_analyze: AsyncMock,
        client: TestClient,
        mock_analyze_response: AnalyzeResponse,
    ) -> None:
        """A valid request should return 200 with an AnalyzeResponse body."""
        mock_analyze.return_value = mock_analyze_response

        response = client.post(
            "/analyze",
            json={
                "financial_text": "Apple reported Q3 revenue of $81.8 billion.",
                "task_type": "summarization",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["selected_model"] == "phi3:mini"
        assert body["routing_decision"] == "slm"
        assert body["confidence_score"] == 0.85
        mock_analyze.assert_awaited_once()

    def test_analyze_empty_text_returns_422(self, client: TestClient) -> None:
        """An empty financial_text should fail Pydantic validation (422)."""
        response = client.post(
            "/analyze",
            json={
                "financial_text": "",
                "task_type": "summarization",
            },
        )
        assert response.status_code == 422

    def test_analyze_missing_fields_returns_422(self, client: TestClient) -> None:
        """Missing required fields should return 422."""
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    @patch("app.api.routes.analyze", new_callable=AsyncMock)
    def test_analyze_internal_error_returns_500(
        self,
        mock_analyze: AsyncMock,
        client: TestClient,
    ) -> None:
        """If the orchestrator raises, the endpoint should return 500."""
        mock_analyze.side_effect = RuntimeError("model timeout")

        response = client.post(
            "/analyze",
            json={
                "financial_text": "Some financial data here.",
                "task_type": "summarization",
            },
        )

        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]
