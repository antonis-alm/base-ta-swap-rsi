import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from almanak.framework.data.market_snapshot import (
    BalanceUnavailableError,
    GasUnavailableError,
    PriceUnavailableError,
    RSIUnavailableError,
)
from almanak.framework.intents import Intent
from almanak.framework.strategies import IntentStrategy, MarketSnapshot, almanak_strategy

logger = logging.getLogger(__name__)


@almanak_strategy(
    name="base_t_a_swap_r_s_i",
    description="RSI-based swap strategy for WETH/USDC on Base",
    version="1.0.0",
    author="Almanak",
    tags=["ta_swap", "rsi", "swap", "base", "uniswap_v3"],
    supported_chains=["base"],
    supported_protocols=["uniswap_v3"],
    intent_types=["SWAP", "HOLD"],
    default_chain="base",
)
class BaseTASwapRSIStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol = str(self.get_config("protocol", "uniswap_v3"))
        self.indicator = str(self.get_config("indicator", "rsi")).lower()
        self.base_token = str(self.get_config("base_token", "WETH"))
        self.quote_token = str(self.get_config("quote_token", "USDC"))

        self.trade_size_usd = Decimal(str(self.get_config("trade_size_usd", "1000")))
        self.max_slippage_bps = int(self.get_config("max_slippage_bps", 30))

        self.rsi_period = int(self.get_config("rsi_period", 14))
        self.rsi_oversold = Decimal(str(self.get_config("rsi_oversold", 30)))
        self.rsi_overbought = Decimal(str(self.get_config("rsi_overbought", 70)))

        self.min_trade_value_usd = Decimal(str(self.get_config("min_trade_value_usd", "10")))
        self.max_gas_ratio = Decimal(str(self.get_config("max_gas_ratio", "0.05")))
        self.force_action = str(self.get_config("force_action", "")).strip().lower()

        self._holding_base = False

    def decide(self, market: MarketSnapshot) -> Intent:
        if self.force_action:
            return self._forced_intent()

        if self.indicator != "rsi":
            return Intent.hold(reason=f"unsupported indicator: {self.indicator}")

        try:
            rsi = market.rsi(self.base_token, period=self.rsi_period)
        except (RSIUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"RSI unavailable: {exc}")

        try:
            quote_balance = market.balance(self.quote_token)
            base_balance = market.balance(self.base_token)
        except (BalanceUnavailableError, ValueError) as exc:
            return Intent.hold(reason=f"Balance unavailable: {exc}")

        if rsi.value <= self.rsi_oversold:
            if quote_balance.balance_usd < self.trade_size_usd:
                return Intent.hold(reason=f"Insufficient {self.quote_token} balance")
            if self.trade_size_usd < self.min_trade_value_usd:
                return Intent.hold(reason="Trade size below minimum")
            if not market.is_trade_worthwhile(
                amount_usd=self.trade_size_usd,
                chain=market.chain,
                max_gas_ratio=self.max_gas_ratio,
            ):
                try:
                    gas_cost = market.estimate_swap_gas_cost_usd(market.chain)
                    return Intent.hold(reason=f"Gas cost too high: ${gas_cost}")
                except (GasUnavailableError, ValueError):
                    return Intent.hold(reason="Gas estimate unavailable")
            return self._buy_intent()

        if rsi.value >= self.rsi_overbought:
            try:
                base_price = market.price(self.base_token)
            except (PriceUnavailableError, ValueError) as exc:
                return Intent.hold(reason=f"Price unavailable: {exc}")

            if base_price <= 0:
                return Intent.hold(reason="Invalid base token price")

            min_base_to_sell = self.trade_size_usd / base_price
            if base_balance.balance < min_base_to_sell:
                return Intent.hold(reason=f"Insufficient {self.base_token} balance")
            if self.trade_size_usd < self.min_trade_value_usd:
                return Intent.hold(reason="Trade size below minimum")
            if not market.is_trade_worthwhile(
                amount_usd=self.trade_size_usd,
                chain=market.chain,
                max_gas_ratio=self.max_gas_ratio,
            ):
                try:
                    gas_cost = market.estimate_swap_gas_cost_usd(market.chain)
                    return Intent.hold(reason=f"Gas cost too high: ${gas_cost}")
                except (GasUnavailableError, ValueError):
                    return Intent.hold(reason="Gas estimate unavailable")
            return self._sell_intent()

        return Intent.hold(
            reason=f"RSI {rsi.value} in neutral zone [{self.rsi_oversold}, {self.rsi_overbought}]"
        )

    def _forced_intent(self) -> Intent:
        if self.force_action == "buy":
            return self._buy_intent()
        if self.force_action == "sell":
            return Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=Decimal(self.max_slippage_bps) / Decimal("10000"),
                protocol=self.protocol,
                chain=self.chain,
            )
        return Intent.hold(reason=f"Unknown force_action: {self.force_action}")

    def _buy_intent(self) -> Intent:
        return Intent.swap(
            from_token=self.quote_token,
            to_token=self.base_token,
            amount_usd=self.trade_size_usd,
            max_slippage=Decimal(self.max_slippage_bps) / Decimal("10000"),
            protocol=self.protocol,
            chain=self.chain,
        )

    def _sell_intent(self) -> Intent:
        return Intent.swap(
            from_token=self.base_token,
            to_token=self.quote_token,
            amount_usd=self.trade_size_usd,
            max_slippage=Decimal(self.max_slippage_bps) / Decimal("10000"),
            protocol=self.protocol,
            chain=self.chain,
        )

    def supports_teardown(self) -> bool:
        return True

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions: list[PositionInfo] = []

        try:
            market = self.create_market_snapshot()
            base_balance = market.balance(self.base_token)
            if base_balance.balance > 0:
                positions.append(
                    PositionInfo(
                        position_type=PositionType.TOKEN,
                        position_id="base_token_exposure",
                        chain=self.chain,
                        protocol=self.protocol,
                        value_usd=Decimal(str(base_balance.balance_usd)),
                        details={
                            "asset": self.base_token,
                            "quote": self.quote_token,
                            "balance": str(base_balance.balance),
                        },
                    )
                )
        except (BalanceUnavailableError, ValueError):
            logger.warning("Unable to query live balance for teardown position summary")

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", "base_t_a_swap_r_s_i"),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        positions = self.get_open_positions()
        if not positions.positions:
            return []

        max_slippage = (
            Decimal("0.03")
            if mode == TeardownMode.HARD
            else Decimal(self.max_slippage_bps) / Decimal("10000")
        )

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

    def on_intent_executed(self, intent, success: bool, result):
        if not success:
            return

        intent_type = getattr(intent, "intent_type", None)
        if not intent_type or intent_type.value != "SWAP":
            return

        if getattr(intent, "to_token", None) == self.base_token:
            self._holding_base = True
        elif getattr(intent, "from_token", None) == self.base_token:
            self._holding_base = False

    def get_persistent_state(self) -> dict[str, Any]:
        return {"holding_base": self._holding_base}

    def load_persistent_state(self, state: dict[str, Any] | None):
        if not state:
            return
        self._holding_base = bool(state.get("holding_base", False))

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "base_t_a_swap_r_s_i",
            "chain": self.chain,
            "base_token": self.base_token,
            "quote_token": self.quote_token,
            "indicator": self.indicator,
            "trade_size_usd": str(self.trade_size_usd),
            "max_slippage_bps": self.max_slippage_bps,
            "holding_base": self._holding_base,
        }
