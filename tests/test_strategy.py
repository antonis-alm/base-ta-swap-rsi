from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from strategy import BaseTASwapRSIStrategy


@pytest.fixture
def config() -> dict:
    return {
        "chain": "base",
        "protocol": "uniswap_v3",
        "indicator": "rsi",
        "base_token": "WETH",
        "quote_token": "USDC",
        "trade_size_usd": 100,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "max_slippage_bps": 30,
        "min_trade_value_usd": "10",
        "max_gas_ratio": "0.05",
        "force_action": "",
    }


@pytest.fixture
def strategy(config: dict) -> BaseTASwapRSIStrategy:
    return BaseTASwapRSIStrategy(
        config=config,
        chain=config["chain"],
        wallet_address="0x" + "1" * 40,
    )


def _balance(balance: str, balance_usd: str) -> MagicMock:
    item = MagicMock()
    item.balance = Decimal(balance)
    item.balance_usd = Decimal(balance_usd)
    return item


def _market(
    *,
    rsi_value: str = "50",
    quote_balance_usd: str = "1000",
    base_balance: str = "1",
    base_balance_usd: str = "2000",
    price: str = "2000",
    worthwhile: bool = True,
) -> MagicMock:
    market = MagicMock()
    market.chain = "base"

    rsi = MagicMock()
    rsi.value = Decimal(rsi_value)
    market.rsi.return_value = rsi

    balances = {
        "USDC": _balance("1000", quote_balance_usd),
        "WETH": _balance(base_balance, base_balance_usd),
    }
    market.balance.side_effect = lambda token: balances[token]
    market.price.return_value = Decimal(price)
    market.is_trade_worthwhile.return_value = worthwhile
    market.estimate_swap_gas_cost_usd.return_value = Decimal("1")
    return market


def test_buy_signal_returns_swap(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="20")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "USDC"
    assert intent.to_token == "WETH"
    assert intent.max_slippage == Decimal("0.003")


def test_sell_signal_returns_swap(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="80", base_balance="1", price="2000")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "WETH"
    assert intent.to_token == "USDC"


def test_neutral_signal_returns_hold(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="50")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"


def test_insufficient_quote_balance_holds(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="20", quote_balance_usd="50")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "Insufficient USDC" in intent.reason


def test_insufficient_base_balance_holds(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="80", base_balance="0.01", price="2000")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "Insufficient WETH" in intent.reason


def test_unworthwhile_trade_holds(strategy: BaseTASwapRSIStrategy):
    market = _market(rsi_value="20", worthwhile=False)

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "Gas cost too high" in intent.reason


def test_rsi_unavailable_holds(strategy: BaseTASwapRSIStrategy):
    market = _market()
    market.rsi.side_effect = ValueError("missing history")

    intent = strategy.decide(market)

    assert intent.intent_type.value == "HOLD"
    assert "RSI unavailable" in intent.reason


def test_force_buy_returns_swap(config: dict):
    config["force_action"] = "buy"
    strategy = BaseTASwapRSIStrategy(config=config, chain="base", wallet_address="0x" + "1" * 40)

    intent = strategy.decide(_market(rsi_value="50"))

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "USDC"


def test_force_sell_returns_swap(config: dict):
    config["force_action"] = "sell"
    strategy = BaseTASwapRSIStrategy(config=config, chain="base", wallet_address="0x" + "1" * 40)

    intent = strategy.decide(_market(rsi_value="50"))

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "WETH"
    assert intent.amount == "all"


def test_teardown_intents_soft_and_hard(strategy: BaseTASwapRSIStrategy, monkeypatch):
    fake_market = MagicMock()
    fake_market.balance.return_value = _balance("0.2", "400")
    monkeypatch.setattr(strategy, "create_market_snapshot", lambda: fake_market)

    from almanak.framework.teardown import TeardownMode

    soft = strategy.generate_teardown_intents(mode=TeardownMode.SOFT)
    hard = strategy.generate_teardown_intents(mode=TeardownMode.HARD)

    assert len(soft) == 1
    assert len(hard) == 1
    assert soft[0].max_slippage == Decimal("0.003")
    assert hard[0].max_slippage == Decimal("0.03")


def test_state_persistence_roundtrip(strategy: BaseTASwapRSIStrategy):
    strategy._holding_base = True
    saved = strategy.get_persistent_state()

    fresh = BaseTASwapRSIStrategy(
        config={
            "chain": "base",
            "indicator": "rsi",
            "base_token": "WETH",
            "quote_token": "USDC",
        },
        chain="base",
        wallet_address="0x" + "2" * 40,
    )
    fresh.load_persistent_state(saved)

    assert fresh.get_persistent_state()["holding_base"] is True
