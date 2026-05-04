import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

from almanak.framework.data.market_snapshot import (
    BalanceUnavailableError,
    GasUnavailableError,
    OHLCVUnavailableError,
    PoolReservesUnavailableError,
    RSIUnavailableError,
    SlippageEstimateUnavailableError,
)
from almanak.framework.intents import Intent
from almanak.framework.strategies import IntentStrategy, MarketSnapshot, almanak_strategy

logger = logging.getLogger(__name__)


class Regime(StrEnum):
    LONG_WETH = "LONG_WETH"
    LONG_USDC = "LONG_USDC"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


@almanak_strategy(
    name="base_t_a_swap_r_s_i",
    description="RSI regime flipper swap strategy for Base WETH/USDC Uniswap V3 0.05%",
    version="1.0.0",
    author="Almanak",
    tags=["rsi", "swap", "uniswap_v3", "base", "regime-flipper"],
    supported_chains=["base"],
    supported_protocols=["uniswap_v3"],
    intent_types=["SWAP", "HOLD"],
    default_chain="base",
)
class BaseTASwapRSIStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol = str(self.get_config("protocol", "uniswap_v3"))
        self.base_token = str(self.get_config("base_token", "WETH"))
        self.quote_token = str(self.get_config("quote_token", "USDC"))
        self.pool_address = str(self.get_config("pool_address", "")).lower()
        self.pool_fee_bps = int(self.get_config("pool_fee_bps", 500))

        self.rsi_period = int(self.get_config("rsi_period", 14))
        self.rsi_timeframe = str(self.get_config("rsi_timeframe", "15m"))
        self.rsi_lower_band = Decimal(str(self.get_config("rsi_lower_band", "45")))
        self.rsi_upper_band = Decimal(str(self.get_config("rsi_upper_band", "55")))

        self.allocation_pct = Decimal(str(self.get_config("allocation_pct", "0.95")))
        self.max_slippage_bps = int(self.get_config("max_slippage_bps", 30))
        self.max_price_impact_bps = int(self.get_config("max_price_impact_bps", 30))
        self.min_expected_output_usd = Decimal(str(self.get_config("min_expected_output_usd", "10")))
        self.min_expected_output_ratio = Decimal(str(self.get_config("min_expected_output_ratio", "0.97")))
        self.min_trade_value_usd = Decimal(str(self.get_config("min_trade_value_usd", "10")))
        self.min_source_balance = Decimal(str(self.get_config("min_source_balance", "0")))
        self.min_pool_tvl_usd = Decimal(str(self.get_config("min_pool_tvl_usd", "0")))
        self.max_gas_ratio = Decimal(str(self.get_config("max_gas_ratio", "0.05")))

        self.cooldown_minutes = int(self.get_config("cooldown_minutes", 5))
        self.max_consecutive_failed_swaps = int(self.get_config("max_consecutive_failed_swaps", 0))
        self.force_action = str(self.get_config("force_action", "")).strip().lower()

        self._prev_rsi: Decimal | None = None
        self._last_processed_candle_ts: str | None = None
        self._signal_regime = Regime.NEUTRAL
        self._portfolio_regime = Regime.UNKNOWN
        self._cooldown_until: datetime | None = None
        self._pending_target_regime: Regime | None = None
        self._consecutive_failed_swaps = 0
        self._last_decision_reason = ""
        self._has_base_exposure = False

    def decide(self, market: MarketSnapshot) -> Intent:
        now = getattr(market, "timestamp", datetime.now(UTC))
        logger.info("decide at %s", now.isoformat())

        if self.force_action:
            return self._forced_intent(market)

        if self.max_consecutive_failed_swaps > 0 and self._consecutive_failed_swaps >= self.max_consecutive_failed_swaps:
            return self._hold("stopped after repeated failed swaps")

        if self._cooldown_until and now < self._cooldown_until:
            return self._hold(f"cooldown active until {self._cooldown_until.isoformat()}")

        closed_candle_ts = self._latest_closed_candle_timestamp(market, now)
        if closed_candle_ts is None:
            return self._hold("no confirmed candle close yet")

        closed_candle_iso = closed_candle_ts.isoformat()
        if self._last_processed_candle_ts == closed_candle_iso:
            return self._hold("candle already processed")

        try:
            rsi_data = market.rsi(self.base_token, period=self.rsi_period, timeframe=self.rsi_timeframe)
        except (RSIUnavailableError, ValueError) as exc:
            return self._hold(f"rsi unavailable: {exc}")

        current_rsi = Decimal(str(rsi_data.value))

        if self._prev_rsi is None:
            self._prev_rsi = current_rsi
            self._last_processed_candle_ts = closed_candle_iso
            return self._hold("seeded initial rsi sample")

        cross_up = self._prev_rsi <= self.rsi_upper_band and current_rsi > self.rsi_upper_band
        cross_down = self._prev_rsi >= self.rsi_lower_band and current_rsi < self.rsi_lower_band

        self._prev_rsi = current_rsi
        self._last_processed_candle_ts = closed_candle_iso

        if not cross_up and not cross_down:
            self._signal_regime = Regime.NEUTRAL
            return self._hold(f"neutral RSI zone: {current_rsi}")

        target_regime = Regime.LONG_WETH if cross_up else Regime.LONG_USDC
        self._signal_regime = target_regime

        if self._portfolio_regime == target_regime:
            return self._hold(f"already in {target_regime.value}")

        source_token = self.quote_token if target_regime == Regime.LONG_WETH else self.base_token
        destination_token = self.base_token if target_regime == Regime.LONG_WETH else self.quote_token

        try:
            source_balance = market.balance(source_token)
        except (BalanceUnavailableError, ValueError) as exc:
            return self._hold(f"balance unavailable for {source_token}: {exc}")

        source_amount = Decimal(str(source_balance.balance)) * self.allocation_pct
        source_value_usd = Decimal(str(source_balance.balance_usd)) * self.allocation_pct

        if source_amount <= self.min_source_balance:
            return self._hold(f"insufficient {source_token} balance")
        if source_value_usd < self.min_trade_value_usd:
            return self._hold(f"trade value ${source_value_usd} below minimum")

        if not self._pool_is_healthy(market):
            return self._hold("pool liquidity unavailable")

        if not self._slippage_checks_pass(market, source_token, destination_token, source_amount, source_value_usd):
            return self._hold("slippage or price-impact checks failed")

        if not market.is_trade_worthwhile(amount_usd=source_value_usd, chain=self.chain, max_gas_ratio=self.max_gas_ratio):
            try:
                gas_cost = market.estimate_swap_gas_cost_usd(self.chain)
                return self._hold(f"gas ${gas_cost} too high for trade ${source_value_usd}")
            except (GasUnavailableError, ValueError):
                return self._hold("gas estimate unavailable")

        self._pending_target_regime = target_regime
        return Intent.swap(
            from_token=source_token,
            to_token=destination_token,
            amount=source_amount,
            max_slippage=Decimal(self.max_slippage_bps) / Decimal("10000"),
            max_price_impact=Decimal(self.max_price_impact_bps) / Decimal("10000"),
            protocol=self.protocol,
            chain=self.chain,
        )

    def _forced_intent(self, market: MarketSnapshot) -> Intent:
        action = self.force_action
        self.force_action = ""

        if action == "buy":
            return self._forced_swap(market, self.quote_token, self.base_token, Regime.LONG_WETH)
        if action == "sell":
            return self._forced_swap(market, self.base_token, self.quote_token, Regime.LONG_USDC)
        return self._hold(f"unknown force_action: {action}")

    def _forced_swap(self, market: MarketSnapshot, from_token: str, to_token: str, target_regime: Regime) -> Intent:
        try:
            source_balance = market.balance(from_token)
        except (BalanceUnavailableError, ValueError) as exc:
            return self._hold(f"forced action balance unavailable: {exc}")

        source_amount = Decimal(str(source_balance.balance)) * self.allocation_pct
        if source_amount <= self.min_source_balance:
            return self._hold(f"forced action insufficient {from_token}")

        self._pending_target_regime = target_regime
        return Intent.swap(
            from_token=from_token,
            to_token=to_token,
            amount=source_amount,
            max_slippage=Decimal(self.max_slippage_bps) / Decimal("10000"),
            max_price_impact=Decimal(self.max_price_impact_bps) / Decimal("10000"),
            protocol=self.protocol,
            chain=self.chain,
        )

    def _pool_is_healthy(self, market: MarketSnapshot) -> bool:
        try:
            pool = market.pool_reserves(self.pool_address, chain=self.chain)
        except (PoolReservesUnavailableError, ValueError):
            return False

        if getattr(pool, "dex", "") != "uniswap_v3":
            return False
        if int(getattr(pool, "fee_tier", 0)) != self.pool_fee_bps:
            return False

        liquidity = getattr(pool, "liquidity", 0) or 0
        if liquidity <= 0:
            return False

        tvl_usd = Decimal(str(getattr(pool, "tvl_usd", "0")))
        if self.min_pool_tvl_usd > 0 and tvl_usd < self.min_pool_tvl_usd:
            return False

        return True

    def _slippage_checks_pass(
        self,
        market: MarketSnapshot,
        source_token: str,
        destination_token: str,
        source_amount: Decimal,
        source_value_usd: Decimal,
    ) -> bool:
        try:
            estimate = market.estimate_slippage(
                token_in=source_token,
                token_out=destination_token,
                amount=source_amount,
                chain=self.chain,
                protocol=self.protocol,
            )
        except (SlippageEstimateUnavailableError, ValueError):
            return False

        if int(getattr(estimate, "effective_slippage_bps", 0)) > self.max_slippage_bps:
            return False
        if int(getattr(estimate, "price_impact_bps", 0)) > self.max_price_impact_bps:
            return False

        recommended_size = Decimal(str(getattr(estimate, "recommended_max_size", source_amount)))
        if recommended_size > 0 and source_amount > recommended_size:
            return False

        effective_slippage = Decimal(int(getattr(estimate, "effective_slippage_bps", 0))) / Decimal("10000")
        expected_output_usd = source_value_usd * (Decimal("1") - effective_slippage)

        if expected_output_usd < self.min_expected_output_usd:
            return False
        if expected_output_usd < source_value_usd * self.min_expected_output_ratio:
            return False

        return True

    def _latest_closed_candle_timestamp(self, market: MarketSnapshot, now: datetime) -> datetime | None:
        ohlcv_fn = getattr(market, "ohlcv", None)
        if not callable(ohlcv_fn):
            return now

        try:
            ohlcv = ohlcv_fn(
                f"{self.base_token}/{self.quote_token}",
                timeframe=self.rsi_timeframe,
                limit=3,
                pool_address=self.pool_address,
            )
        except (OHLCVUnavailableError, ValueError):
            return None

        if ohlcv is None or ohlcv.empty or "timestamp" not in ohlcv.columns:
            return None

        candles = ohlcv.sort_values("timestamp")
        if len(candles) < 2:
            return None

        last_ts = candles.iloc[-1]["timestamp"]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)

        timeframe = _timeframe_to_timedelta(self.rsi_timeframe)
        if now < last_ts + timeframe:
            closed_ts = candles.iloc[-2]["timestamp"]
            if closed_ts.tzinfo is None:
                closed_ts = closed_ts.replace(tzinfo=UTC)
            return closed_ts

        return last_ts

    def on_intent_executed(self, intent, success: bool, result) -> None:
        if getattr(intent.intent_type, "value", str(intent.intent_type)) != "SWAP":
            return

        if success:
            from_token = str(getattr(intent, "from_token", ""))
            to_token = str(getattr(intent, "to_token", ""))

            if self._pending_target_regime is not None:
                self._portfolio_regime = self._pending_target_regime
                self._has_base_exposure = self._pending_target_regime == Regime.LONG_WETH
            elif from_token == self.base_token and to_token == self.quote_token:
                self._portfolio_regime = Regime.LONG_USDC
                self._has_base_exposure = False
            elif from_token == self.quote_token and to_token == self.base_token:
                self._portfolio_regime = Regime.LONG_WETH
                self._has_base_exposure = True

            self._pending_target_regime = None
            self._consecutive_failed_swaps = 0
            self._cooldown_until = datetime.now(UTC) + timedelta(minutes=self.cooldown_minutes)
            return

        self._pending_target_regime = None
        self._consecutive_failed_swaps += 1

    def _hold(self, reason: str) -> Intent:
        self._last_decision_reason = reason
        logger.info("hold: %s", reason)
        return Intent.hold(reason=reason)

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "base_t_a_swap_r_s_i",
            "chain": self.chain,
            "signal_regime": self._signal_regime.value,
            "portfolio_regime": self._portfolio_regime.value,
            "consecutive_failed_swaps": self._consecutive_failed_swaps,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "last_processed_candle_ts": self._last_processed_candle_ts,
            "last_decision_reason": self._last_decision_reason,
            "has_base_exposure": self._has_base_exposure,
        }

    def get_persistent_state(self) -> dict[str, Any]:
        return {
            "prev_rsi": str(self._prev_rsi) if self._prev_rsi is not None else None,
            "last_processed_candle_ts": self._last_processed_candle_ts,
            "signal_regime": self._signal_regime.value,
            "portfolio_regime": self._portfolio_regime.value,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "pending_target_regime": self._pending_target_regime.value if self._pending_target_regime else None,
            "consecutive_failed_swaps": self._consecutive_failed_swaps,
            "last_decision_reason": self._last_decision_reason,
            "has_base_exposure": self._has_base_exposure,
        }

    def load_persistent_state(self, state: dict[str, Any]) -> None:
        if not state:
            return

        prev_rsi = state.get("prev_rsi")
        self._prev_rsi = Decimal(prev_rsi) if prev_rsi is not None else None
        self._last_processed_candle_ts = state.get("last_processed_candle_ts")
        self._signal_regime = Regime(state.get("signal_regime", Regime.NEUTRAL.value))
        self._portfolio_regime = Regime(state.get("portfolio_regime", Regime.UNKNOWN.value))

        cooldown_until = state.get("cooldown_until")
        self._cooldown_until = datetime.fromisoformat(cooldown_until) if cooldown_until else None

        pending_regime = state.get("pending_target_regime")
        self._pending_target_regime = Regime(pending_regime) if pending_regime else None

        self._consecutive_failed_swaps = int(state.get("consecutive_failed_swaps", 0))
        self._last_decision_reason = str(state.get("last_decision_reason", ""))
        self._has_base_exposure = bool(state.get("has_base_exposure", False))

    def get_open_positions(self):
        from datetime import datetime

        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions: list[PositionInfo] = []
        if self._has_base_exposure:
            positions.append(
                PositionInfo(
                    position_type=PositionType.TOKEN,
                    position_id="base_ta_swap_rsi_weth",
                    chain=self.chain,
                    protocol=self.protocol,
                    value_usd=Decimal("0"),
                    details={"asset": self.base_token, "quote": self.quote_token},
                )
            )

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", self.STRATEGY_NAME),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode=None, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        has_base_balance = self._has_base_exposure
        if market is not None:
            try:
                base_balance = market.balance(self.base_token)
                has_base_balance = Decimal(str(base_balance.balance)) > self.min_source_balance
            except (BalanceUnavailableError, ValueError):
                pass

        if not has_base_balance:
            return []

        soft_slippage = max(Decimal(self.max_slippage_bps) / Decimal("10000"), Decimal("0.01"))
        max_slippage = Decimal("0.03") if mode == TeardownMode.HARD else soft_slippage

        return [
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )
        ]


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    if timeframe.endswith("m"):
        return timedelta(minutes=int(timeframe[:-1]))
    if timeframe.endswith("h"):
        return timedelta(hours=int(timeframe[:-1]))
    if timeframe.endswith("d"):
        return timedelta(days=int(timeframe[:-1]))
    raise ValueError(f"unsupported timeframe: {timeframe}")
