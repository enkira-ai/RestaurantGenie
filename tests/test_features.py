from unittest.mock import MagicMock, patch
from tests.conftest import FAKE_OSM_RESPONSE


def test_fetch_pois_for_bbox_returns_typed_list():
    from src.features import fetch_pois_for_bbox
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_OSM_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.features.requests.post", return_value=mock_resp):
        pois = fetch_pois_for_bbox(36.10, -115.20, 36.25, -115.10)
    assert len(pois) == 6
    types = {p["type"] for p in pois}
    assert types == {"bar", "restaurant", "office", "hotel", "transit", "school"}
    for p in pois:
        assert "lat" in p and "lon" in p and "type" in p
    pizza = next(p for p in pois if p["type"] == "restaurant")
    assert pizza["cuisine"] == "pizza"
