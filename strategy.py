import logging
import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from almanak.framework.data.timeframes import OHLCVTimeframe
from almanak.framework.intents import AnyIntent, Intent
from almanak.framework.market import MarketSnapshot
from almanak.framework.market.errors import (
    BalanceUnavailableError,
    OHLCVUnavailableError,
    PriceUnavailableError,
    VolatilityUnavailableError,
)
from almanak.framework.strategies import DecideResult, IntentStrategy, almanak_strategy

logger = logging.getLogger(__name__)


@almanak_strategy(
    name="guarded_pancakeswap_v3_lp",
    description="Guarded PEPE/BSC-USD PancakeSwap V3 concentrated-liquidity LP",
    version="1.0.0",
    author="Almanak",
    tags=["concentrated-liquidity", "pancakeswap-v3", "guarded"],
    supported_chains=["bsc"],
    supported_protocols=["pancakeswap_v3"],
    intent_types=["LP_OPEN", "LP_CLOSE", "SWAP", "HOLD"],
    default_chain="bsc",
    quote_asset="USD",
)
class GuardedPancakeswapV3LpStrategy(IntentStrategy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        def config_value(key: str, default: Any) -> Any:
            return self.config.get(key, default) if isinstance(self.config, dict) else getattr(self.config, key, default)

        self.protocol = str(config_value("protocol", "pancakeswap_v3"))
        self.pool_address = str(config_value("pool_address", ""))
        self.fee_tier = int(config_value("fee_tier", 0))
        self.token0_symbol = str(config_value("token0_symbol", ""))
        self.token0_address = str(config_value("token0_address", ""))
        self.token0_decimals = int(config_value("token0_decimals", 18))
        self.token1_symbol = str(config_value("token1_symbol", ""))
        self.token1_address = str(config_value("token1_address", ""))
        self.token1_decimals = int(config_value("token1_decimals", 18))
        self.max_lp_allocation = Decimal(str(config_value("max_lp_allocation_pct", 35))) / Decimal("100")
        self.reserve = Decimal(str(config_value("reserve_pct", 65))) / Decimal("100")
        self.max_portfolio_pepe = Decimal(str(config_value("max_portfolio_pepe_pct", 50))) / Decimal("100")
        self.range_width = Decimal(str(config_value("range_width_pct", 20))) / Decimal("100")
        self.recenter_interval = timedelta(hours=int(config_value("recenter_interval_hours", 24)))
        self.max_entry_hourly_vol = Decimal(str(config_value("max_one_hour_realized_volatility_pct", 12))) / Decimal("100")
        self.range_escape_duration = timedelta(minutes=int(config_value("range_escape_minutes", 45)))
        self.volatility_exit_multiplier = Decimal(str(config_value("volatility_exit_multiplier", 2)))
        self.max_swap_price_impact = Decimal(str(config_value("max_swap_price_impact_bps", 75))) / Decimal("10000")
        self.max_slippage = Decimal(str(config_value("max_slippage_bps", 100))) / Decimal("10000")
        self.max_rebalances_per_day = int(config_value("max_rebalances_per_day", 1))
        self.force_action = str(config_value("force_action", "") or "").lower()
        self.force_position_id = config_value("force_position_id", None)

        if self.protocol != "pancakeswap_v3" or self.chain != "bsc":
            raise ValueError("This strategy is restricted to PancakeSwap V3 on BSC")
        if self.pool_address.lower() != "0xbba8f85c3ceddf73db4de17d31608d640eaea416" or self.fee_tier != 2500:
            raise ValueError("Configured pool must be the approved PEPE/BSC-USD PancakeSwap V3 fee-2500 pool")
        if (self.token0_symbol, self.token0_address.lower(), self.token0_decimals) != (
            "PEPE", "0x25d887ce7a35172c62febfd67a1856f20faebb00", 18
        ):
            raise ValueError("Configured token0 must be the approved PEPE record")
        if (self.token1_symbol, self.token1_address.lower(), self.token1_decimals) != (
            "BSC-USD", "0x55d398326f99059ff775485246999027b3197955", 18
        ):
            raise ValueError("Configured token1 must be the approved BSC-USD record")
        if self.max_lp_allocation + self.reserve > Decimal("1") or self.range_width <= 0:
            raise ValueError("LP allocation/reserve or range configuration is invalid")

        self._position_id: str | None = None
        self._range_lower: Decimal | None = None
        self._range_upper: Decimal | None = None
        self._opened_at: datetime | None = None
        self._range_escape_started_at: datetime | None = None
        self._rebalances_day: date | None = None
        self._rebalances_today = 0
        self._pending_recenter = False

    def decide(self, market: MarketSnapshot) -> DecideResult:
        if self.force_action:
            return self._forced_intent(market)

        try:
            pepe_price = Decimal(str(market.price(self.token0_address)))
            stable_price = Decimal(str(market.price(self.token1_address)))
            pepe_balance = market.balance(self.token0_address)
            stable_balance = market.balance(self.token1_address)
            hourly_vol = self._hourly_volatility(market)
        except (PriceUnavailableError, BalanceUnavailableError, OHLCVUnavailableError, VolatilityUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"Required market data unavailable: {exc}")

        if pepe_price <= 0 or stable_price <= 0:
            return Intent.hold(reason="Required market price is non-positive")

        now = self._market_time(market)
        self._reset_daily_counter(now.date())
        if self._position_id is not None:
            return self._manage_position(market, now, hourly_vol)
        return self._enter_or_recenter(pepe_price, stable_price, pepe_balance, stable_balance, hourly_vol)

    def _hourly_volatility(self, market: MarketSnapshot) -> Decimal:
        candles = market.ohlcv(
            self.token0_address,
            timeframe=OHLCVTimeframe.ONE_HOUR,
            limit=24,
            pool_address=self.pool_address,
        )
        closes = [Decimal(str(value)) for value in candles["close"].dropna().tolist()]
        if len(closes) < 24:
            raise OHLCVUnavailableError(
                self.token0_address,
                "requires 24 one-hour close observations for the trailing-day volatility gate",
            )
        if any(value <= 0 for value in closes):
            raise ValueError("OHLCV close observations must be positive")
        returns = [math.log(float(current / previous)) for previous, current in zip(closes, closes[1:])]
        if len(returns) < 2:
            raise OHLCVUnavailableError(self.token0_address, "insufficient one-hour returns")
        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        return Decimal(str(math.sqrt(variance)))

    def _manage_position(self, market: MarketSnapshot, now: datetime, hourly_vol: Decimal) -> DecideResult:
        try:
            position = market.lp_position_value(
                self._position_id,
                self.protocol,
                pool_address=self.pool_address,
                token0_symbol=self.token0_symbol,
                token1_symbol=self.token1_symbol,
            )
        except ValueError as exc:
            return Intent.hold(reason=f"LP position state unavailable: {exc}")
        if position is None:
            return Intent.hold(reason="LP position state unavailable; refusing blind close")

        if hourly_vol > self.max_entry_hourly_vol * self.volatility_exit_multiplier:
            return self._close_intent(self._position_id)

        if not bool(position.in_range):
            if self._range_escape_started_at is None:
                self._range_escape_started_at = now
                return Intent.hold(reason="LP range escape observed; waiting for sustained confirmation")
            if now - self._range_escape_started_at < self.range_escape_duration:
                return Intent.hold(reason="LP range escape has not persisted for configured duration")
            if self._rebalances_today >= self.max_rebalances_per_day:
                return Intent.hold(reason="LP range escape confirmed but daily rebalance cap reached")
            self._pending_recenter = True
            return self._close_intent(self._position_id)

        self._range_escape_started_at = None
        if self._opened_at is not None and now - self._opened_at >= self.recenter_interval:
            if self._rebalances_today >= self.max_rebalances_per_day:
                return Intent.hold(reason="LP recenter interval reached but daily rebalance cap reached")
            self._pending_recenter = True
            return self._close_intent(self._position_id)
        return Intent.hold(reason=f"LP position {self._position_id} remains in range")

    def _enter_or_recenter(self, pepe_price: Decimal, stable_price: Decimal, pepe_balance: Any, stable_balance: Any, hourly_vol: Decimal) -> DecideResult:
        if hourly_vol > self.max_entry_hourly_vol:
            return Intent.hold(reason="One-hour realized PEPE volatility exceeds entry limit")

        pepe_usd = Decimal(str(pepe_balance.balance_usd))
        stable_usd = Decimal(str(stable_balance.balance_usd))
        total_usd = pepe_usd + stable_usd
        if total_usd <= 0:
            return Intent.hold(reason="No priced PEPE/BSC-USD capital available")

        maximum_pepe_usd = total_usd * self.max_portfolio_pepe
        if pepe_usd > maximum_pepe_usd:
            return self._swap_intent(self.token0_address, self.token1_address, pepe_usd - maximum_pepe_usd)

        lp_budget = min(total_usd * self.max_lp_allocation, total_usd * (Decimal("1") - self.reserve))
        leg_usd = lp_budget / Decimal("2")
        if lp_budget <= 0:
            return Intent.hold(reason="LP allocation is zero after reserve requirement")

        if pepe_usd < leg_usd:
            required = leg_usd - pepe_usd
            if pepe_usd + required > maximum_pepe_usd or stable_usd - required < leg_usd:
                return Intent.hold(reason="Cannot acquire PEPE LP leg within reserve and concentration limits")
            return self._swap_intent(self.token1_address, self.token0_address, required)
        if stable_usd < leg_usd:
            required = leg_usd - stable_usd
            if pepe_usd - required < leg_usd:
                return Intent.hold(reason="Cannot acquire BSC-USD LP leg while preserving LP inventory")
            return self._swap_intent(self.token0_address, self.token1_address, required)

        return self._open_intent(pepe_price, stable_price, leg_usd)

    def _open_intent(self, pepe_price: Decimal, stable_price: Decimal, leg_usd: Decimal) -> AnyIntent:
        lower = (pepe_price / stable_price) * (Decimal("1") - self.range_width / Decimal("2"))
        upper = (pepe_price / stable_price) * (Decimal("1") + self.range_width / Decimal("2"))
        return Intent.lp_open(
            pool=self.pool_address,
            amount0=leg_usd / pepe_price,
            amount1=leg_usd / stable_price,
            range_lower=lower,
            range_upper=upper,
            protocol=self.protocol,
            chain=self.chain,
            protocol_params={"fee_tier": self.fee_tier},
            max_slippage=self.max_slippage,
            require_two_sided_minimums=True,
        )

    def _swap_intent(self, from_token: str, to_token: str, amount_usd: Decimal) -> AnyIntent:
        return Intent.swap(
            from_token=from_token,
            to_token=to_token,
            amount_usd=amount_usd,
            max_slippage=self.max_slippage,
            max_price_impact=self.max_swap_price_impact,
            protocol=self.protocol,
            chain=self.chain,
            swap_params={"fee_tier": self.fee_tier},
        )

    def _close_intent(self, position_id: str) -> AnyIntent:
        return Intent.lp_close(
            position_id=position_id,
            pool=self.pool_address,
            collect_fees=True,
            protocol=self.protocol,
            chain=self.chain,
            protocol_params={"fee_tier": self.fee_tier},
        )

    def _forced_intent(self, market: MarketSnapshot) -> DecideResult:
        if self.force_action == "close":
            position_id = str(self.force_position_id or self._position_id or "")
            if not position_id:
                raise ValueError("force_action=close requires force_position_id or an open position")
            return self._close_intent(position_id)
        if self.force_action == "rebalance_swap":
            return self._swap_intent(self.token0_address, self.token1_address, Decimal("1"))
        if self.force_action == "open":
            pepe_price = Decimal(str(market.price(self.token0_address)))
            stable_price = Decimal(str(market.price(self.token1_address)))
            pepe_balance = market.balance(self.token0_address)
            stable_balance = market.balance(self.token1_address)
            if pepe_price <= 0 or stable_price <= 0:
                raise ValueError("force_action=open requires positive token prices")
            pepe_usd = Decimal(str(pepe_balance.balance_usd))
            stable_usd = Decimal(str(stable_balance.balance_usd))
            leg_usd = min(pepe_usd, stable_usd, (pepe_usd + stable_usd) * self.max_lp_allocation / Decimal("2"))
            if leg_usd <= 0:
                raise ValueError("force_action=open requires funded PEPE and BSC-USD balances")
            return self._open_intent(pepe_price, stable_price, leg_usd)
        raise ValueError(f"Unknown force_action: {self.force_action!r}")

    def on_intent_executed(self, intent: AnyIntent, success: bool, result: Any) -> None:
        if not success:
            return
        intent_type = getattr(getattr(intent, "intent_type", None), "value", "")
        if intent_type == "LP_OPEN":
            position_id = getattr(result, "position_id", None)
            if position_id is None:
                return
            self._position_id = str(position_id)
            self._range_lower = Decimal(str(intent.range_lower))
            self._range_upper = Decimal(str(intent.range_upper))
            self._opened_at = datetime.now(UTC)
            self._range_escape_started_at = None
            self._pending_recenter = False
        elif intent_type == "LP_CLOSE":
            if self._pending_recenter:
                today = datetime.now(UTC).date()
                self._reset_daily_counter(today)
                self._rebalances_today += 1
            self._position_id = None
            self._range_lower = None
            self._range_upper = None
            self._range_escape_started_at = None

    def get_persistent_state(self) -> dict[str, Any]:
        return {
            "position_id": self._position_id,
            "range_lower": str(self._range_lower) if self._range_lower is not None else None,
            "range_upper": str(self._range_upper) if self._range_upper is not None else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "range_escape_started_at": self._range_escape_started_at.isoformat() if self._range_escape_started_at else None,
            "rebalances_day": self._rebalances_day.isoformat() if self._rebalances_day else None,
            "rebalances_today": self._rebalances_today,
            "pending_recenter": self._pending_recenter,
        }

    def load_persistent_state(self, state: dict[str, Any]) -> None:
        self._position_id = str(state["position_id"]) if state.get("position_id") is not None else None
        self._range_lower = Decimal(state["range_lower"]) if state.get("range_lower") else None
        self._range_upper = Decimal(state["range_upper"]) if state.get("range_upper") else None
        self._opened_at = datetime.fromisoformat(state["opened_at"]) if state.get("opened_at") else None
        self._range_escape_started_at = datetime.fromisoformat(state["range_escape_started_at"]) if state.get("range_escape_started_at") else None
        self._rebalances_day = date.fromisoformat(state["rebalances_day"]) if state.get("rebalances_day") else None
        self._rebalances_today = int(state.get("rebalances_today", 0))
        self._pending_recenter = bool(state.get("pending_recenter", False))

    def get_teardown_profile(self):
        from almanak.framework.teardown import TeardownAssetPolicy, TeardownProfile

        return TeardownProfile(
            natural_exit_assets=[self.token0_symbol, self.token1_symbol],
            original_entry_assets=[self.token0_symbol, self.token1_symbol],
            recommended_target=self.token1_symbol,
            estimated_steps=1,
            chains_involved=[self.chain],
            has_lp_positions=True,
            preferred_asset_policy=TeardownAssetPolicy.KEEP_OUTPUTS,
        )

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions = []
        if self._position_id is not None:
            details: dict[str, Any] = {
                "pool": self.pool_address,
                "fee_tier": self.fee_tier,
                "token0": self.token0_symbol,
                "token1": self.token1_symbol,
            }
            value_usd = Decimal("0")
            try:
                value = self.create_market_snapshot().lp_position_value(
                    self._position_id,
                    self.protocol,
                    pool_address=self.pool_address,
                    token0_symbol=self.token0_symbol,
                    token1_symbol=self.token1_symbol,
                )
                if value is None:
                    raise ValueError("unmeasured LP position")
                value_usd = Decimal(str(value.total_usd))
            except (PriceUnavailableError, ValueError):
                details["value_usd_unknown"] = True
                details["valuation_status"] = "no_path"
            positions.append(PositionInfo(
                position_type=PositionType.LP,
                position_id=self._position_id,
                chain=self.chain,
                protocol=self.protocol,
                value_usd=value_usd,
                details=details,
            ))
        return TeardownPositionSummary(
            deployment_id=getattr(self, "deployment_id", "guarded_pancakeswap_v3_lp"),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode: Any, market: MarketSnapshot | None = None) -> list[AnyIntent]:
        return [self._close_intent(self._position_id)] if self._position_id is not None else []

    def _market_time(self, market: MarketSnapshot) -> datetime:
        timestamp = market.timestamp
        return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)

    def _reset_daily_counter(self, current_day: date) -> None:
        if self._rebalances_day != current_day:
            self._rebalances_day = current_day
            self._rebalances_today = 0
