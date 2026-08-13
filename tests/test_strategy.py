import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from almanak.framework.market.errors import OHLCVUnavailableError
from almanak.framework.market.models import TokenBalance
from almanak.framework.market.testing import seeded
from almanak.framework.teardown import TeardownMode
from strategy import GuardedPancakeswapV3LpStrategy

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _config(**overrides):
    config = json.loads((Path(__file__).parent.parent / "config.json").read_text())
    config.update(overrides)
    return config


def _strategy(**overrides):
    config = _config(**overrides)
    return GuardedPancakeswapV3LpStrategy(
        config=config,
        chain="bsc",
        wallet_address="0x" + "1" * 40,
    )


def _market(
    *,
    pepe_usd=Decimal("500"),
    stable_usd=Decimal("500"),
    timestamp=T0,
    hourly_vol=Decimal("0.05"),
    in_range=True,
):
    snapshot = seeded(
        chain="bsc",
        wallet_address="0x" + "1" * 40,
        prices={
            "0x25d887ce7a35172c62febfd67a1856f20faebb00": Decimal("0.01"),
            "0x55d398326f99059ff775485246999027b3197955": Decimal("1"),
        },
        balances={
            "0x25d887ce7a35172c62febfd67a1856f20faebb00": TokenBalance(symbol="PEPE", balance=pepe_usd / Decimal("0.01"), balance_usd=pepe_usd),
            "0x55d398326f99059ff775485246999027b3197955": TokenBalance(symbol="BSC-USD", balance=stable_usd, balance_usd=stable_usd),
        },
        timestamp=timestamp,
    )
    snapshot.ohlcv = lambda *args, **kwargs: pd.DataFrame({"close": [Decimal("1")] * 24})
    snapshot.lp_position_value = lambda *args, **kwargs: SimpleNamespace(
        in_range=in_range,
        total_usd=Decimal("350"),
    )
    return snapshot


def _kind(intent):
    return intent.intent_type.value


class TestCapitalAndEntryGuards:
    def test_open_respects_allocation_reserve_and_exact_pool(self):
        strategy = _strategy()
        intent = strategy.decide(_market())

        assert _kind(intent) == "LP_OPEN"
        assert intent.pool == "0xbba8f85c3ceddf73db4de17d31608d640eaea416"
        assert intent.protocol == "pancakeswap_v3"
        assert intent.protocol_params == {"fee_tier": 2500}
        assert intent.amount0 == Decimal("50000")
        assert intent.amount1 == Decimal("500")
        assert intent.max_slippage == Decimal("0.01")
        assert intent.require_two_sided_minimums is True

    def test_hourly_volatility_reads_exact_pool_with_24_candles(self):
        strategy = _strategy()
        market = _market()
        observed = {}

        def ohlcv(token, **kwargs):
            observed["token"] = token
            observed.update(kwargs)
            closes = [Decimal("1")]
            for index in range(23):
                closes.append(closes[-1] * (Decimal("1.01") if index % 2 == 0 else Decimal("0.99")))
            return pd.DataFrame({"close": closes})

        market.ohlcv = ohlcv
        assert strategy._hourly_volatility(market) > 0
        assert observed == {
            "token": "0x25d887ce7a35172c62febfd67a1856f20faebb00",
            "timeframe": "1h",
            "limit": 24,
            "pool_address": "0xbba8f85c3ceddf73db4de17d31608d640eaea416",
        }

    def test_hourly_volatility_fails_closed_when_fewer_than_24_closes(self):
        strategy = _strategy()
        market = _market()
        market.ohlcv = lambda *args, **kwargs: pd.DataFrame({"close": [Decimal("1")] * 23})
        with pytest.raises(OHLCVUnavailableError, match="requires 24 one-hour close observations"):
            strategy._hourly_volatility(market)

    def test_pepe_concentration_cap_emits_guarded_correction_swap(self):
        strategy = _strategy()
        intent = strategy.decide(_market(pepe_usd=Decimal("600"), stable_usd=Decimal("400")))

        assert _kind(intent) == "SWAP"
        assert intent.from_token == "0x25d887ce7a35172c62febfd67a1856f20faebb00"
        assert intent.to_token == "0x55d398326f99059ff775485246999027b3197955"
        assert intent.amount_usd == Decimal("100")
        assert intent.max_slippage == Decimal("0.01")
        assert intent.max_price_impact == Decimal("0.0075")
        assert intent.swap_params == {"fee_tier": 2500}

    def test_entry_uses_smaller_available_leg_without_exact_balance(self):
        strategy = _strategy()
        intent = strategy.decide(_market(pepe_usd=Decimal("2.49"), stable_usd=Decimal("2.51")))

        assert _kind(intent) == "LP_OPEN"
        assert intent.amount0 == Decimal("249")
        assert intent.amount1 == Decimal("2.49")

    def test_entry_caps_available_leg_at_configured_lp_budget(self):
        strategy = _strategy(max_lp_allocation_pct=80)
        intent = strategy.decide(_market(pepe_usd=Decimal("2.49"), stable_usd=Decimal("2.51")))

        assert _kind(intent) == "LP_OPEN"
        assert intent.amount0 == Decimal("200")
        assert intent.amount1 == Decimal("2")

    @pytest.mark.parametrize(
        ("volatility", "expected"),
        [(Decimal("0.12"), "LP_OPEN"), (Decimal("0.120001"), "HOLD")],
    )
    def test_entry_hourly_volatility_boundary(self, volatility, expected):
        strategy = _strategy()
        strategy._hourly_volatility = lambda market: volatility
        assert _kind(strategy.decide(_market(hourly_vol=volatility))) == expected


class TestPositionRiskControls:
    def _open_strategy(self):
        strategy = _strategy()
        strategy._position_id = "123"
        strategy._opened_at = T0
        return strategy

    def test_range_escape_requires_full_configured_duration(self):
        strategy = self._open_strategy()
        assert _kind(strategy.decide(_market(timestamp=T0, in_range=False))) == "HOLD"
        assert _kind(strategy.decide(_market(timestamp=T0 + timedelta(minutes=44, seconds=59), in_range=False))) == "HOLD"
        intent = strategy.decide(_market(timestamp=T0 + timedelta(minutes=45), in_range=False))
        assert _kind(intent) == "LP_CLOSE"
        assert intent.pool == strategy.pool_address
        assert intent.collect_fees is True

    @pytest.mark.parametrize(
        ("volatility", "expected"),
        [(Decimal("0.24"), "HOLD"), (Decimal("0.240001"), "LP_CLOSE")],
    )
    def test_volatility_exit_boundary(self, volatility, expected):
        strategy = self._open_strategy()
        strategy._hourly_volatility = lambda market: volatility
        assert _kind(strategy.decide(_market(hourly_vol=volatility))) == expected

    def test_daily_rebalance_cap_blocks_scheduled_recenter(self):
        strategy = self._open_strategy()
        strategy._opened_at = T0 - timedelta(hours=24)
        strategy._rebalances_day = T0.date()
        strategy._rebalances_today = 1
        intent = strategy.decide(_market(timestamp=T0, in_range=True))
        assert _kind(intent) == "HOLD"
        assert "daily rebalance cap" in intent.reason


class TestLifecycleAndTeardown:
    def test_force_actions_cover_each_non_hold_intent(self):
        swap = _strategy(force_action="rebalance_swap").decide(_market())
        forced = _strategy(force_action="open")
        open_intent = forced.decide(_market(pepe_usd=Decimal("600"), stable_usd=Decimal("400")))
        forced.on_intent_executed(open_intent, True, SimpleNamespace(position_id=456))
        forced.force_action = "close"
        close_intent = forced.decide(_market())

        assert _kind(swap) == "SWAP"
        assert swap.swap_params == {"fee_tier": 2500}
        assert _kind(open_intent) == "LP_OPEN"
        assert open_intent.protocol_params == {"fee_tier": 2500}
        assert _kind(close_intent) == "LP_CLOSE"
        assert close_intent.position_id == "456"
        assert close_intent.protocol_params == {"fee_tier": 2500}

    def test_forced_close_accepts_explicit_position_id(self):
        intent = _strategy(force_action="close", force_position_id="789").decide(_market())
        assert _kind(intent) == "LP_CLOSE"
        assert intent.position_id == "789"
        assert intent.protocol_params == {"fee_tier": 2500}

    def test_open_close_state_and_asset_preserving_teardown(self):
        strategy = _strategy()
        open_intent = strategy.decide(_market())
        strategy.on_intent_executed(open_intent, True, SimpleNamespace(position_id=123))
        assert strategy._position_id == "123"

        profile = strategy.get_teardown_profile()
        assert profile.natural_exit_assets == ["PEPE", "BSC-USD"]
        intents = strategy.generate_teardown_intents(TeardownMode.SOFT)
        assert len(intents) == 1
        assert _kind(intents[0]) == "LP_CLOSE"
        assert intents[0].collect_fees is True
        assert intents[0].protocol_params == {"fee_tier": 2500}

        strategy.on_intent_executed(intents[0], True, SimpleNamespace())
        assert strategy._position_id is None

    def test_persistent_state_round_trip(self):
        strategy = _strategy()
        strategy._position_id = "123"
        strategy._range_lower = Decimal("0.009")
        strategy._range_upper = Decimal("0.011")
        strategy._opened_at = T0
        strategy._range_escape_started_at = T0 + timedelta(minutes=1)
        strategy._rebalances_day = T0.date()
        strategy._rebalances_today = 1
        strategy._pending_recenter = True

        restored = _strategy()
        restored.load_persistent_state(strategy.get_persistent_state())
        assert restored.get_persistent_state() == strategy.get_persistent_state()
