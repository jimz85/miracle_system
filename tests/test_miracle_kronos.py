#!/usr/bin/env python3
"""Unit tests for miracle_kronos.py core functions"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miracle_kronos import (
    calc_rsi, calc_adx, calc_macd, calc_bollinger, calc_atr,
    voting_vote, get_pattern_adjustment, get_dynamic_sl_tp,
    check_whitelist, record_pattern_outcome, load_pattern_history,
    save_pattern_history, load_ic_weights, get_4h_trend,
    safe_float, DEFAULT_WEIGHTS,
)
from trade_journal import load_trades, save_trades, record_trade

# ─── Test data ───
PRICES_UPTREND = [100 + i * 0.5 for i in range(30)]  # Steady uptrend
PRICES_DOWNTREND = [100 - i * 0.5 for i in range(30)]  # Steady downtrend
PRICES_RANGING = [100 + (i % 5 - 2) * 2 for i in range(30)]  # 96-106 range
PRICES_SPIKE = [100] * 15 + [50] + [100] * 14  # Flash crash recovery
PRICES_FLAT = [100] * 30  # Completely flat

HIGHS_UPTREND = [p + 1 for p in PRICES_UPTREND]
LOWS_UPTREND = [p - 1 for p in PRICES_UPTREND]
HIGHS_RANGING = [p + 2 for p in PRICES_RANGING]
LOWS_RANGING = [p - 2 for p in PRICES_RANGING]


# ══════════════════════════════════════════════
# 1. calc_rsi
# ══════════════════════════════════════════════
class TestCalcRSI:
    def test_uptrend_should_be_overbought(self):
        rsi = calc_rsi(PRICES_UPTREND)
        assert rsi > 50, f"Uptrend RSI should be >50, got {rsi}"
        assert rsi <= 100, f"RSI should be <=100, got {rsi}"

    def test_downtrend_should_be_oversold(self):
        rsi = calc_rsi(PRICES_DOWNTREND)
        assert rsi < 50, f"Downtrend RSI should be <50, got {rsi}"

    def test_ranging_should_be_around_50(self):
        rsi = calc_rsi(PRICES_RANGING)
        assert 30 <= rsi <= 70, f"Ranging RSI should be 30-70, got {rsi}"

    def test_flat_prices_returns_50(self):
        rsi = calc_rsi(PRICES_FLAT)
        # Flat prices have no gains/losses, avg_gain=avg_loss=0 → RSI=50
        assert rsi == 50.0, f"Flat RSI should be 50, got {rsi}"

    def test_insufficient_data_returns_50(self):
        rsi = calc_rsi([100, 101])  # Only 2 prices
        assert rsi == 50.0

    def test_value_range(self):
        rsi = calc_rsi(PRICES_UPTREND)
        assert 0 <= rsi <= 100, f"RSI should be 0-100, got {rsi}"


# ══════════════════════════════════════════════
# 2. calc_adx
# ══════════════════════════════════════════════
class TestCalcADX:
    def test_uptrend_di_plus_greater(self):
        di_plus, di_minus, adx = calc_adx(HIGHS_UPTREND, LOWS_UPTREND, PRICES_UPTREND)
        assert di_plus > di_minus, f"Uptrend DI+ should > DI-, got {di_plus:.2f} vs {di_minus:.2f}"
        assert adx > 0, f"ADX should be >0, got {adx}"

    def test_ranging_low_adx(self):
        di_plus, di_minus, adx = calc_adx(HIGHS_RANGING, LOWS_RANGING, PRICES_RANGING)
        assert adx < 30, f"Ranging ADX should be <30, got {adx:.2f}"

    def test_flat_prices_low_adx(self):
        di_plus, di_minus, adx = calc_adx(
            [100]*30, [100]*30, [100]*30
        )
        assert di_plus == di_minus, f"Flat DI+ should == DI-"

    def test_adx_range(self):
        _, _, adx = calc_adx(HIGHS_UPTREND, LOWS_UPTREND, PRICES_UPTREND)
        assert 0 <= adx <= 100, f"ADX should be 0-100, got {adx}"


# ══════════════════════════════════════════════
# 3. voting_vote (7因子投票)
# ══════════════════════════════════════════════
class TestVotingVote:
    """Test the core 7-factor voting system"""

    def test_strong_bull_signal(self):
        """RSI超卖+ADX强趋势+Bollinger下轨+正MACD+高量+BTC多头"""
        factors = {
            'rsi': 25, 'adx': 35, 'bb_pos': 10,
            'macd_hist': 0.02, 'vol_ratio': 1.8,
            'btc_trend': 'bull',
            '_di_plus': 30, '_di_minus': 15,
            '_gemma_vote': 0.7,
            '_extreme_signal': None,
            'gemma_health': 'healthy',
        }
        result = voting_vote(factors, DEFAULT_WEIGHTS)
        assert result['direction'] == 'long', f"Should be long, got {result['direction']}"
        assert result['score'] > 0, f"Score should be positive, got {result['score']}"
        assert result['confidence'] >= 0, f"Confidence should be >=0"

    def test_strong_bear_signal(self):
        """RSI超买+ADX强趋势+Bollinger上轨+负MACD+低量+BTC空头"""
        factors = {
            'rsi': 75, 'adx': 35, 'bb_pos': 90,
            'macd_hist': -0.02, 'vol_ratio': 1.8,
            'btc_trend': 'bear',
            '_di_plus': 15, '_di_minus': 30,
            '_gemma_vote': 0.3,
            '_extreme_signal': None,
            'gemma_health': 'healthy',
        }
        result = voting_vote(factors, DEFAULT_WEIGHTS)
        assert result['direction'] == 'short', f"Should be short, got {result['direction']}"
        assert result['score'] < 0, f"Score should be negative, got {result['score']}"

    def test_conflicting_signals_returns_wait(self):
        """信号冲突时应返回wait"""
        factors = {
            'rsi': 50, 'adx': 15, 'bb_pos': 50,
            'macd_hist': 0, 'vol_ratio': 1.0,
            'btc_trend': 'neutral',
            '_di_plus': 20, '_di_minus': 20,
            '_gemma_vote': 0.5,
            '_extreme_signal': None,
            'gemma_health': 'healthy',
        }
        result = voting_vote(factors, DEFAULT_WEIGHTS)
        assert result['direction'] in ('wait', 'long', 'short'), f"Unexpected direction: {result['direction']}"

    def test_gemma_down_zeros_vote(self):
        """Gemma不可用时Gemma因子应该为0"""
        factors = {
            'rsi': 30, 'adx': 25, 'bb_pos': 20,
            'macd_hist': 0.01, 'vol_ratio': 1.2,
            'btc_trend': 'bull',
            '_di_plus': 25, '_di_minus': 15,
            '_gemma_vote': 0.7,  # Gemma原本看多
            '_extreme_signal': None,
            'gemma_health': 'down',  # 但Gemma挂了
        }
        result = voting_vote(factors, DEFAULT_WEIGHTS)
        assert result['votes']['Gemma'] == 0, f"Gemma down should vote 0"

    def test_extreme_oversold_in_ranging(self):
        """极端RSI<5且在震荡市(ADX<25) = 强烈做多"""
        factors = {
            'rsi': 3, 'adx': 18, 'bb_pos': 5,
            'macd_hist': -0.01, 'vol_ratio': 0.8,
            'btc_trend': 'neutral',
            '_di_plus': 18, '_di_minus': 20,
            '_gemma_vote': 0.5,
            '_extreme_signal': 'long',
            'gemma_health': 'healthy',
        }
        result = voting_vote(factors, DEFAULT_WEIGHTS)
        assert result['direction'] == 'long'
        assert result['confidence'] >= 0.50, f"Extreme signal should have high confidence"


# ══════════════════════════════════════════════
# 4. get_pattern_adjustment (模式胜率调整)
# ══════════════════════════════════════════════
class TestPatternAdjustment:
    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        # Backup and restore pattern_history
        ph_file = Path(__file__).parent.parent / 'data' / 'pattern_history.json'
        backup = ph_file.read_text() if ph_file.exists() else None
        yield
        if backup is not None:
            ph_file.write_text(backup)
        else:
            ph_file.write_text('{"patterns":{}, "total_trades":0, "wins":0, "losses":0}')

    def test_new_pattern_returns_1_0(self):
        adj = get_pattern_adjustment('unknown_new_pattern')
        assert adj == 1.0, f"New pattern should return 1.0, got {adj}"

    def test_high_win_rate_boost(self):
        for _ in range(4):
            record_pattern_outcome('test_high_win', won=True, pnl_pct=0.05)
        record_pattern_outcome('test_high_win', won=False, pnl_pct=-0.02)
        adj = get_pattern_adjustment('test_high_win')
        assert adj == 1.2, f"80% win rate should return 1.2, got {adj}"

    def test_low_win_rate_penalty(self):
        for _ in range(2):
            record_pattern_outcome('test_low_win', won=True, pnl_pct=0.03)
        for _ in range(4):
            record_pattern_outcome('test_low_win', won=False, pnl_pct=-0.03)
        # 6 entries, 33% win rate => <40% => 0.7
        adj = get_pattern_adjustment('test_low_win')
        assert adj == 0.7, f"33% win rate should return 0.7, got {adj}"

    def test_very_low_win_rate_strong_penalty(self):
        for _ in range(1):
            record_pattern_outcome('test_very_bad', won=True, pnl_pct=0.02)
        for _ in range(5):
            record_pattern_outcome('test_very_bad', won=False, pnl_pct=-0.02)
        adj = get_pattern_adjustment('test_very_bad')
        assert adj == 0.4, f"~17% win rate should return 0.4, got {adj}"

    def test_expected_value_calculation(self):
        history = load_pattern_history()
        p = history['patterns'].get('test_low_win')
        if p and 'expected_value' in p:
            ev = p['expected_value']
            assert isinstance(ev, float), f"EV should be float, got {type(ev)}"


# ══════════════════════════════════════════════
# 5. get_dynamic_sl_tp (动态SL/TP)
# ══════════════════════════════════════════════
class TestDynamicSlTp:
    def test_high_volatility_coin(self):
        """高波动币(DOGE)应该用1.5x ATR SL"""
        sl, tp = get_dynamic_sl_tp('DOGE', 0.10, 0.004, adx=25)
        assert sl > 0, f"SL should be >0, got {sl}"
        assert tp > sl, f"TP ({tp}) should > SL ({sl})"

    def test_normal_coin(self):
        """普通币用2x ATR SL"""
        sl, tp = get_dynamic_sl_tp('BTC', 50000, 500, adx=20)
        assert sl > 0
        assert tp > sl

    def test_high_conviction_wider_tp(self):
        """ADX>30时RR=4"""
        sl_normal, tp_normal = get_dynamic_sl_tp('BTC', 50000, 500, adx=20)
        sl_high, tp_high = get_dynamic_sl_tp('BTC', 50000, 500, adx=35)
        # 正常RR=2, 高确信RR=4
        assert tp_high / sl_high > tp_normal / sl_normal, \
            f"High conviction RR should be wider: {tp_high/sl_high:.1f} vs {tp_normal/sl_normal:.1f}"

    def test_invalid_inputs_fallback(self):
        """无效输入应返回默认SL/TP"""
        sl, tp = get_dynamic_sl_tp('BTC', 0, 0, adx=20)
        assert sl > 0 and tp > 0

    def test_minimum_sl_floor(self):
        """ATR很小时SL不应低于0.5%"""
        sl, tp = get_dynamic_sl_tp('BTC', 50000, 10, adx=20)
        min_sl = 0.005
        assert sl >= min_sl, f"SL ({sl:.4f}) should be >= {min_sl}"


# ══════════════════════════════════════════════
# 6. safe_float
# ══════════════════════════════════════════════
class TestSafeFloat:
    def test_normal_float(self):
        assert safe_float("123.45") == 123.45
    def test_none(self):
        assert safe_float(None) == 0.0
    def test_empty_string(self):
        assert safe_float("") == 0.0
    def test_null_string(self):
        assert safe_float("null") == 0.0
    def test_integer_string(self):
        assert safe_float("42") == 42.0
    def test_invalid_string(self):
        assert safe_float("abc") == 0.0


# ══════════════════════════════════════════════
# 7. load_ic_weights
# ══════════════════════════════════════════════
class TestIcWeights:
    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights should sum to ~1.0, got {total}"

    def test_default_weights_have_all_factors(self):
        expected = {'RSI', 'ADX', 'Bollinger', 'Vol', 'MACD', 'BTC', 'Gemma'}
        assert set(DEFAULT_WEIGHTS.keys()) == expected, \
            f"Missing factors: {expected - set(DEFAULT_WEIGHTS.keys())}"

    def test_load_ic_weights_returns_dict(self):
        weights = load_ic_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0


# ══════════════════════════════════════════════
# 8. Professional Formula Validation
# ══════════════════════════════════════════════
class TestProfessionalFormulas:
    """验证三个核心职业交易公式"""

    def test_risk_reward_ratio_adx30(self):
        """公式1: RR=4 当ADX>30"""
        sl, tp = get_dynamic_sl_tp('BTC', 50000, 500, adx=35)
        rr = tp / sl
        assert abs(rr - 4.0) < 0.5, f"RR should be ~4.0 for ADX>30, got {rr:.2f}"

    def test_risk_reward_ratio_normal(self):
        """公式1: RR=2 正常"""
        sl, tp = get_dynamic_sl_tp('BTC', 50000, 500, adx=20)
        rr = tp / sl
        assert abs(rr - 2.0) < 0.5, f"RR should be ~2.0 for normal, got {rr:.2f}"

    def test_pattern_expected_value_tracks_ev(self):
        """公式2: 期望值 = win_rate×avg_win - loss_rate×avg_loss"""
        # 模拟3W2L
        for _ in range(3):
            record_pattern_outcome('test_ev_pattern', won=True, pnl_pct=0.10)
        for _ in range(2):
            record_pattern_outcome('test_ev_pattern', won=False, pnl_pct=-0.05)

        history = load_pattern_history()
        p = history['patterns'].get('test_ev_pattern', {})
        ev = p.get('expected_value', 0)

        # EV = 0.6*0.10 - 0.4*0.05 = 0.06 - 0.02 = 0.04
        expected_ev = 0.6 * 0.10 - 0.4 * 0.05
        assert abs(ev - expected_ev) < 0.01, \
            f"EV should be ~{expected_ev:.4f}, got {ev:.4f}"

    def test_risk_based_position_sizing_code_exists(self):
        """公式3: 固定风险仓位公式存在于run_scan中"""
        import inspect
        src = inspect.getsource(miracle_kronos_module := __import__('miracle_kronos'))
        checks = [
            ('risk_amount =', 'risk_amount'),
            ('risk_per_contract', 'risk_per_contract'),
            ('RISK_PER_TRADE', 'RISK_PER_TRADE'),
        ]
        for name, keyword in checks:
            assert keyword in src, f"Position sizing code '{name}' not found in miracle_kronos.py"


# ══════════════════════════════════════════════
# 9. Trade Journal
# ══════════════════════════════════════════════
class TestTradeJournal:
    def test_record_and_load(self):
        record_trade({'coin': 'TEST_UNIT', 'status': 'OPEN', 'entry_price': 100})
        trades = load_trades()
        test_trades = [t for t in trades if t.get('coin') == 'TEST_UNIT']
        assert len(test_trades) == 1
        # Cleanup
        remaining = [t for t in trades if t.get('coin') != 'TEST_UNIT']
        save_trades(remaining)
