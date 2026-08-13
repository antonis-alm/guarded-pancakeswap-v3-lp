"""Tests for the guarded PancakeSwap V3 LP dashboard."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock


class _LPDashboardConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_dashboard(monkeypatch):
    templates = ModuleType("almanak.framework.dashboard.templates")
    templates.LPDashboardConfig = _LPDashboardConfig
    templates.prepare_lp_session_state = MagicMock()
    templates.render_lp_dashboard = MagicMock()
    monkeypatch.setitem(sys.modules, "almanak.framework.dashboard.templates", templates)
    sys.modules.pop("dashboard.ui", None)
    return importlib.import_module("dashboard.ui")


def test_dashboard_uses_validated_pancakeswap_pool(monkeypatch):
    dashboard = _load_dashboard(monkeypatch)

    assert dashboard.FEE_TIER == 2500
    assert dashboard.DASHBOARD_CONFIG.protocol == "pancakeswap_v3"
    assert dashboard.DASHBOARD_CONFIG.chain == "bsc"
    assert dashboard.DASHBOARD_CONFIG.pool_address == "0xbba8f85c3ceddf73db4de17d31608d640eaea416"
    assert dashboard.DASHBOARD_CONFIG.fee_tier == "0.25%"
    assert dashboard.DASHBOARD_CONFIG.token0 == "PEPE"
    assert dashboard.DASHBOARD_CONFIG.token1 == "BSC-USD"


def test_dashboard_prepares_live_state_then_renders_template(monkeypatch):
    dashboard = _load_dashboard(monkeypatch)
    prepared_state = {"position_id": "42", "is_active": True}
    prepare = MagicMock(return_value=prepared_state)
    render = MagicMock()
    monkeypatch.setattr(dashboard, "prepare_lp_session_state", prepare)
    monkeypatch.setattr(dashboard, "render_lp_dashboard", render)
    client = MagicMock()
    state = {"position_id": "stale"}
    strategy_config = {"pool_address": dashboard.POOL_ADDRESS, "fee_tier": 2500}

    dashboard.render_custom_dashboard("deployment-1", strategy_config, client, state)

    prepare.assert_called_once_with(
        client,
        session_state=state,
        config=dashboard.DASHBOARD_CONFIG,
        deployment_id="deployment-1",
    )
    render.assert_called_once_with(
        "deployment-1",
        strategy_config,
        prepared_state,
        dashboard.DASHBOARD_CONFIG,
        api_client=client,
    )
