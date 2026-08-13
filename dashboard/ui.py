"""Dashboard for the guarded PEPE/BSC-USD PancakeSwap V3 LP strategy."""

from typing import Any

from almanak.framework.dashboard.templates import (
    LPDashboardConfig,
    prepare_lp_session_state,
    render_lp_dashboard,
)

POOL_ADDRESS = "0xbba8f85c3ceddf73db4de17d31608d640eaea416"
FEE_TIER = 2500

DASHBOARD_CONFIG = LPDashboardConfig(
    protocol="pancakeswap_v3",
    token0="PEPE",
    token1="BSC-USD",
    fee_tier="0.25%",
    chain="bsc",
    pool_address=POOL_ADDRESS,
    token0_address="0x25d887ce7a35172c62febfd67a1856f20faebb00",
    token1_address="0x55d398326f99059ff775485246999027b3197955",
)


def render_custom_dashboard(
    deployment_id: str,
    strategy_config: dict[str, Any],
    api_client: Any,
    session_state: dict[str, Any],
) -> None:
    live_state = prepare_lp_session_state(
        api_client,
        session_state=session_state,
        config=DASHBOARD_CONFIG,
        deployment_id=deployment_id,
    )
    render_lp_dashboard(
        deployment_id,
        strategy_config,
        live_state,
        DASHBOARD_CONFIG,
        api_client=api_client,
    )
