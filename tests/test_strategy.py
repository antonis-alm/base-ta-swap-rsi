from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from almanak.framework.intents import Intent
from almanak.framework.strategies import RSIData, TokenBalance

from strategy import BaseTASwapRSIStrategy, Regime


@pytest.fixture
def config() -> dict:
    return {
        "chain": "base",
        "protocol": "uniswap_v3",
        "base_token": "WETH",
        "quote_token": "USDC",
        "pool_address": "0xd0b53d9277642d899df5c87a3966a349a798f224",
        "pool_fee_bps": 500,
        "rsi_period": 14,
        "rsi_timeframe": "15m",
        "rsi_lower_band": 45,
        "rsi_upper_band": 55,
        "allocation_pct": "0.95",
        "max_slippage_bps": 30,
        "max_price_impact_bps": 30,
        "min_expected_output_usd": "10",
        "min_expected_output_ratio": "0.97",
        "min_trade_value_usd": "10",
        "min_source_balance": "0.0001",
        "min_pool_tvl_usd": "0",
        "max_gas_ratio": "0.05",
        "cooldown_minutes": 5,
        "max_consecutive_failed_swaps": 3,
        "force_action": "",
    }


@pytest.fixture
def strategy(config: dict) -> BaseTASwapRSIStrategy:
    return BaseTASwapRSIStrategy(config=config, chain="base", wallet_address="0x" + "1" * 40)


def _market(
    *,
    now: datetime,
    rsi_value: Decimal,
    balances: dict[str, TokenBalance],
    pool_fee_bps: int = 500,
    liquidity: int = 1,
    effective_slippage_bps: int = 10,
    price_impact_bps: int = 10,
    worthwhile: bool = True,
) -> MagicMock:
    market = MagicMock()
    market.timestamp = now

    candles = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 45, tzinfo=UTC),
            ],
            "close": [2300.0, 2310.0, 2320.0],
        }
    )
    market.ohlcv.return_value = candles

    market.rsi.return_value = RSIData(value=rsi_value, period=14)

    def _balance_side_effect(token: str):
        return balances[token]

    market.balance.side_effect = _balance_side_effect

    market.pool_reserves.return_value = SimpleNamespace(
        dex="uniswap_v3",
        fee_tier=pool_fee_bps,
        liquidity=liquidity,
        tvl_usd=Decimal("1000000"),
    )

    market.estimate_slippage.return_value = SimpleNamespace(
        effective_slippage_bps=effective_slippage_bps,
        price_impact_bps=price_impact_bps,
        recommended_max_size=Decimal("100000"),
    )

    market.is_trade_worthwhile.return_value = worthwhile
    market.estimate_swap_gas_cost_usd.return_value = Decimal("1")
    return market


def _default_balances() -> dict[str, TokenBalance]:
    return {
        "USDC": TokenBalance(symbol="USDC", balance=Decimal("1000"), balance_usd=Decimal("1000")),
        "WETH": TokenBalance(symbol="WETH", balance=Decimal("1"), balance_usd=Decimal("2300")),
    }


def test_seeds_first_rsi_sample(strategy: BaseTASwapRSIStrategy):
    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("50"), balances=_default_balances())

    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert strategy._prev_rsi == Decimal("50")


def test_skips_already_processed_candle(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("50")
    strategy._last_processed_candle_ts = datetime(2026, 1, 1, 0, 45, tzinfo=UTC).isoformat()

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("56"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "already processed" in result.reason


def test_neutral_zone_holds(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("50")

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("52"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert strategy._signal_regime == Regime.NEUTRAL


def test_cross_above_upper_flips_to_weth(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("56"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "SWAP"
    assert result.from_token == "USDC"
    assert result.to_token == "WETH"


def test_cross_below_lower_flips_to_usdc(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("46")

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("44"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "SWAP"
    assert result.from_token == "WETH"
    assert result.to_token == "USDC"


def test_avoids_repeat_swap_when_in_target_regime(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")
    strategy._portfolio_regime = Regime.LONG_WETH

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("56"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "already in LONG_WETH" in result.reason


def test_cooldown_blocks_flip(strategy: BaseTASwapRSIStrategy):
    strategy._cooldown_until = datetime.now(UTC) + timedelta(minutes=2)

    market = _market(now=datetime.now(UTC), rsi_value=Decimal("56"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "cooldown active" in result.reason


def test_holds_when_insufficient_balance(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")
    balances = _default_balances()
    balances["USDC"] = TokenBalance(symbol="USDC", balance=Decimal("0"), balance_usd=Decimal("0"))

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("56"), balances=balances)
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "insufficient USDC" in result.reason


def test_holds_on_pool_fee_mismatch(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")

    market = _market(
        now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC),
        rsi_value=Decimal("56"),
        balances=_default_balances(),
        pool_fee_bps=3000,
    )
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "pool liquidity unavailable" in result.reason


def test_swaps_when_pool_reserves_method_unavailable(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")

    market = _market(
        now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC),
        rsi_value=Decimal("56"),
        balances=_default_balances(),
    )
    market.pool_reserves = None

    result = strategy.decide(market)

    assert result.intent_type.value == "SWAP"
    assert result.from_token == "USDC"
    assert result.to_token == "WETH"


def test_holds_when_slippage_too_high(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")

    market = _market(
        now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC),
        rsi_value=Decimal("56"),
        balances=_default_balances(),
        effective_slippage_bps=200,
    )
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "slippage or price-impact checks failed" in result.reason


def test_holds_when_trade_not_worth_gas(strategy: BaseTASwapRSIStrategy):
    strategy._prev_rsi = Decimal("54")

    market = _market(
        now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC),
        rsi_value=Decimal("56"),
        balances=_default_balances(),
        worthwhile=False,
    )
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "gas" in result.reason


def test_failure_stop_condition(strategy: BaseTASwapRSIStrategy):
    strategy._consecutive_failed_swaps = 3

    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("56"), balances=_default_balances())
    result = strategy.decide(market)

    assert result.intent_type.value == "HOLD"
    assert "repeated failed swaps" in result.reason


def test_force_action_buy_and_sell(strategy: BaseTASwapRSIStrategy):
    market = _market(now=datetime(2026, 1, 1, 1, 10, tzinfo=UTC), rsi_value=Decimal("50"), balances=_default_balances())

    strategy.force_action = "buy"
    buy = strategy.decide(market)
    assert buy.intent_type.value == "SWAP"
    assert buy.from_token == "USDC"
    assert strategy.force_action == ""

    strategy.force_action = "sell"
    sell = strategy.decide(market)
    assert sell.intent_type.value == "SWAP"
    assert sell.from_token == "WETH"
    assert strategy.force_action == ""


def test_on_intent_executed_updates_state(strategy: BaseTASwapRSIStrategy):
    swap = Intent.swap("USDC", "WETH", amount=Decimal("10"))

    strategy._pending_target_regime = Regime.LONG_WETH
    strategy._consecutive_failed_swaps = 2

    strategy.on_intent_executed(swap, success=True, result=SimpleNamespace())
    assert strategy._portfolio_regime == Regime.LONG_WETH
    assert strategy._has_base_exposure is True
    assert strategy._consecutive_failed_swaps == 0
    assert strategy._cooldown_until is not None

    teardown_swap = Intent.swap("WETH", "USDC", amount="all")
    strategy.on_intent_executed(teardown_swap, success=True, result=SimpleNamespace())
    assert strategy._portfolio_regime == Regime.LONG_USDC
    assert strategy._has_base_exposure is False

    strategy.on_intent_executed(swap, success=False, result=SimpleNamespace())
    assert strategy._consecutive_failed_swaps == 1


def test_persistent_state_roundtrip(strategy: BaseTASwapRSIStrategy, config: dict):
    strategy._prev_rsi = Decimal("53")
    strategy._last_processed_candle_ts = datetime(2026, 1, 1, 0, 45, tzinfo=UTC).isoformat()
    strategy._signal_regime = Regime.LONG_WETH
    strategy._portfolio_regime = Regime.LONG_USDC
    strategy._cooldown_until = datetime(2026, 1, 1, 1, 15, tzinfo=UTC)
    strategy._pending_target_regime = Regime.LONG_WETH
    strategy._consecutive_failed_swaps = 2
    strategy._last_decision_reason = "test"

    state = strategy.get_persistent_state()

    restored = BaseTASwapRSIStrategy(config=config, chain="base", wallet_address="0x" + "2" * 40)
    restored.load_persistent_state(state)

    assert restored.get_persistent_state() == state


def test_teardown_and_open_positions(strategy: BaseTASwapRSIStrategy):
    assert strategy.generate_teardown_intents() == []

    strategy._has_base_exposure = True
    teardown_intents = strategy.generate_teardown_intents()
    assert len(teardown_intents) == 1
    assert teardown_intents[0].intent_type.value == "SWAP"
    assert teardown_intents[0].amount == "all"

    summary = strategy.get_open_positions()
    assert len(summary.positions) == 1
