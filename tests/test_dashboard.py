from unittest.mock import patch

from dashboard.ui import _build_dashboard_config, render_custom_dashboard


def test_build_dashboard_config_uses_strategy_values():
    config = _build_dashboard_config({"rsi_period": 21, "rsi_upper_band": 55, "rsi_lower_band": 45})

    assert config.indicator_name == "RSI"
    assert config.indicator_period == 21
    assert config.upper_threshold == 55
    assert config.lower_threshold == 45


def test_build_dashboard_config_falls_back_to_standard_rsi_keys():
    config = _build_dashboard_config({"rsi_period": 10, "rsi_overbought": 68, "rsi_oversold": 32})

    assert config.indicator_period == 10
    assert config.upper_threshold == 68
    assert config.lower_threshold == 32


def test_render_custom_dashboard_calls_ta_template():
    with patch("dashboard.ui.render_ta_dashboard") as render_ta_dashboard:
        strategy_config = {"rsi_period": 14, "rsi_upper_band": 55, "rsi_lower_band": 45}
        session_state = {"rsi_value": 51}

        render_custom_dashboard(
            strategy_id="base_t_a_swap_r_s_i",
            strategy_config=strategy_config,
            api_client=None,
            session_state=session_state,
        )

        render_ta_dashboard.assert_called_once()
        args = render_ta_dashboard.call_args.args
        assert args[0] == "base_t_a_swap_r_s_i"
        assert args[1] == strategy_config
        assert args[2] == session_state
        assert args[3].indicator_name == "RSI"
        assert args[3].upper_threshold == 55
        assert args[3].lower_threshold == 45
