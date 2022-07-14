from fastapi.testclient import TestClient
from main import app, VERSION

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText": "Root Endpoint for Microbes API",
        "statusCode": 200,
        "version": VERSION,
    }


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusCode": 200,
        "statusText": "Microbes API Version Fetch Successful",
        "version": VERSION,
    }


def test_infer():
    response = client.get("/infer")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }