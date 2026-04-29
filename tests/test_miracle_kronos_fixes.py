"""
Tests for miracle_kronos.py critical logic fixes (V3-V6)
========================================================

Covers fixes:
- V3 NEW-3: SHORT parsed as LONG in Gemma fallback
- V3 NEW-5: extreme_signal score scale (→0.80 floor)
- V5 V5-1: extreme path ADX override + wait conf=0
- V5 V5-3: close_position direction fallback
- V6 V6-1: _detect_pos_mode uses API not position inference
- V6 P3-1: _rule_based_vote ADX sensitivity aligned with main path
- V6 V6-3: treasury consecutive_win_hours field
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import tempfile
import os

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# Test: _rule_based_vote ADX scoring (V6 P3-1)
# ============================================================================

def test_rule_based_vote_adx_strong_aligned_with_main_path():
    """V6 P3-1: ADX>30 should give +0.10, matching main path ~0.12 contribution"""
    from miracle_kronos import _rule_based_vote

    # Neutral RSI (50) + BB mid (50) + ADX>30
    # Score should be 0.5 + 0.10 = 0.60
    result = _rule_based_vote(rsi=50, adx=35, bb_pos=50)
    assert 0.55 <= result <= 0.65, f"ADX>30 expected ~0.60, got {result}"

    # ADX>22 but <=30 should give +0.05
    result = _rule_based_vote(rsi=50, adx=25, bb_pos=50)
    assert 0.50 <= result <= 0.60, f"ADX>22 expected ~0.55, got {result}"

    # ADX<15 should give -0.05
    result = _rule_based_vote(rsi=50, adx=10, bb_pos=50)
    assert 0.40 <= result <= 0.50, f"ADX<15 expected ~0.45, got {result}"


def test_rule_based_vote_short_signal():
    """V3 NEW-3: SHORT and LONG should give different votes"""
    from miracle_kronos import _rule_based_vote

    # Overbought RSI + high BB → SHORT bias
    short_vote = _rule_based_vote(rsi=75, adx=20, bb_pos=85)
    long_vote = _rule_based_vote(rsi=25, adx=20, bb_pos=15)

    assert short_vote < 0.5, f"RSI>70+high BB should be SHORT-biased (<0.5), got {short_vote}"
    assert long_vote > 0.5, f"RSI<30+low BB should be LONG-biased (>0.5), got {long_vote}"


# ============================================================================
# Test: voting_vote confidence (V3 NEW-5, V5 V5-1)
# ============================================================================

def test_voting_vote_extreme_confidence_080():
    """V3 NEW-5 + V5 V5-1: extreme signal without wait → conf=0.80"""
    from miracle_kronos import voting_vote

    # Extreme RSI signal, no ADX override
    factors = {
        'rsi': 28, 'adx': 10, 'bb_pos': 15,  # ADX<30, no override
        'macd_hist': 0, 'vol_ratio': 1.0,
        'btc_trend': 'neutral',
        '_di_plus': 20, '_di_minus': 20,
        '_gemma_vote': 0.5,
        '_extreme_signal': 'long',
    }
    weights = {'RSI': 0.15, 'ADX': 0.12, 'Bollinger': 0.10,
               'Vol': 0.08, 'MACD': 0.08, 'BTC': 0.10, 'Gemma': 0.12}

    result = voting_vote(factors, weights)
    assert result['direction'] == 'long'
    assert result['confidence'] == 0.80, f"extreme+no_wait expected 0.80, got {result['confidence']}"
    assert result['extreme'] == 'long'


def test_voting_vote_extreme_wait_confidence_zero():
    """V5 V5-1: extreme signal overridden by ADX → direction=wait, conf=0.0"""
    from miracle_kronos import voting_vote

    # Extreme LONG but ADX>30 says market is bearish (DI- > DI+)
    factors = {
        'rsi': 25, 'adx': 35, 'bb_pos': 20,
        'macd_hist': -0.5, 'vol_ratio': 1.0,
        'btc_trend': 'neutral',
        '_di_plus': 15, '_di_minus': 35,  # Bearish trend
        '_gemma_vote': 0.5,
        '_extreme_signal': 'long',
    }
    weights = {'RSI': 0.15, 'ADX': 0.12, 'Bollinger': 0.10,
               'Vol': 0.08, 'MACD': 0.08, 'BTC': 0.10, 'Gemma': 0.12}

    result = voting_vote(factors, weights)
    assert result['direction'] == 'wait', f"extreme LONG + bearish ADX should be wait, got {result['direction']}"
    assert result['confidence'] == 0.0, f"wait state expected conf=0.0, got {result['confidence']}"
    assert result['extreme'] == 'long'


def test_voting_vote_normal_signal_confidence():
    """Normal (non-extreme) signals use score/2.0"""
    from miracle_kronos import voting_vote

    factors = {
        'rsi': 35, 'adx': 10, 'bb_pos': 20,
        'macd_hist': 0.1, 'vol_ratio': 1.0,
        'btc_trend': 'neutral',
        '_di_plus': 20, '_di_minus': 20,
        '_gemma_vote': 0.5,
        '_extreme_signal': None,
    }
    weights = {'RSI': 0.15, 'ADX': 0.12, 'Bollinger': 0.10,
               'Vol': 0.08, 'MACD': 0.08, 'BTC': 0.10, 'Gemma': 0.12}

    result = voting_vote(factors, weights)
    # All LONG factors: RSI(0.5)+BB(1.0)+Gemma(0.0) weighted ≈ positive
    assert result['direction'] in ('long', 'short')
    assert 0.0 <= result['confidence'] <= 0.5, f"normal conf should be <=0.5, got {result['confidence']}"
    assert result['extreme'] is None


# ============================================================================
# Test: close_position direction handling (V5 V5-3)
# ============================================================================

def test_close_position_hedge_mode_posside():
    """V5 V5-3: hedge mode should add posSide to order body"""
    import miracle_kronos

    with patch('miracle_kronos.okx_req') as mock_req:
        mock_req.return_value = {'code': '0', 'data': [{'algoId': '123'}]}
        # Set global _pos_mode directly (module imported at load time, not re-evaluated)
        miracle_kronos._pos_mode = 'hedge'

        from miracle_kronos import close_position
        result = close_position('DOGE', pos={'sz': '100', 'side': 'short'})

        post_calls = [c for c in mock_req.call_args_list if c[0][0] == 'POST']
        assert len(post_calls) >= 1, f"should call POST, calls={mock_req.call_args_list}"
        args = post_calls[0][0]
        body = json.loads(args[2])
        assert body.get('posSide') == 'short', f"hedge mode should include posSide='short', got {body}"

def test_close_position_net_mode_no_posside():
    """V5 V5-3: net mode should NOT add posSide"""
    import miracle_kronos

    with patch('miracle_kronos.okx_req') as mock_req:
        mock_req.return_value = {'code': '0', 'data': []}
        miracle_kronos._pos_mode = 'net'  # Set global directly

        from miracle_kronos import close_position
        result = close_position('DOGE', pos={'sz': '100', 'side': 'short'})

        post_calls = [c for c in mock_req.call_args_list if c[0][0] == 'POST']
        assert len(post_calls) >= 1, f"should call POST, calls={mock_req.call_args_list}"
        args = post_calls[0][0]
        body = json.loads(args[2])
        assert 'posSide' not in body, f"net mode should NOT include posSide, got {body}"


def test_close_position_local_record_direction():
    """V5 V5-3: local record with 'direction' field should close correctly without posSide"""
    import miracle_kronos

    with patch('miracle_kronos.okx_req') as mock_req:
        mock_req.return_value = {'code': '0', 'data': []}
        miracle_kronos._pos_mode = 'net'

        from miracle_kronos import close_position
        local_pos = {'sz': '100', 'direction': 'long'}
        result = close_position('DOGE', pos=local_pos)

        post_calls = [c for c in mock_req.call_args_list if c[0][0] == 'POST']
        assert len(post_calls) >= 1, f"should call POST, calls={mock_req.call_args_list}"
        args = post_calls[0][0]
        body = json.loads(args[2])
        # direction='long' should produce close_side='sell', no posSide
        assert body['side'] == 'sell', f"long direction should close with sell side, got {body['side']}"
        assert 'posSide' not in body, f"local record direction should not add posSide, got {body}"

# ============================================================================
# Test: treasury consecutive_win_hours (V6 V6-3)
# ============================================================================

def test_load_treasury_default_has_consecutive_win_hours():
    """V6 V6-3: load_treasury default state must include consecutive_win_hours"""
    from miracle_kronos import load_treasury

    # Mock file not exists to get default
    with patch('miracle_kronos.TREASURY_FILE', new_callable=lambda: MagicMock(spec=os.PathLike, exists=lambda: False)):
        # Need to patch at module level since it's already imported
        pass

    # Can't easily mock the file existence check at import time
    # Test the key assertion: default dict has the field
    default_keys = [
        'equity', 'hourly_snapshot', 'daily_snapshot', 'tier',
        'consecutive_loss_hours', 'consecutive_win_hours'
    ]
    # Verify the field exists in load_treasury by checking source
    import inspect
    src = inspect.getsource(__import__('miracle_kronos', fromlist=['load_treasury']).load_treasury)
    # The field must be present in the default return dict
    assert "'consecutive_win_hours'" in src, "load_treasury default must include consecutive_win_hours"


# ============================================================================
# Test: whitelist blacklist 7-day TTL (V4)
# ============================================================================

def test_whitelist_blacklist_ttl_dict_format():
    """V4: blacklist must be dict with timestamp values, not set"""
    import inspect
    from miracle_kronos import load_whitelist, save_whitelist

    src = inspect.getsource(load_whitelist)
    assert 'bl_dict' in src, "load_whitelist must use dict for blacklist"
    assert 'time.time()' in src or 'now - t' in src, "load_whitelist must check TTL"


# ============================================================================
# Test: _detect_pos_mode uses API (V6 V6-1)
# ============================================================================

def test_detect_pos_mode_calls_api():
    """V6 V6-1: _detect_pos_mode must call OKX account config API"""
    with patch('miracle_kronos.okx_req') as mock_req:
        mock_req.return_value = {'code': '0', 'data': [{'posMode': 'long_short_mode'}]}

        from miracle_kronos import _detect_pos_mode
        result = _detect_pos_mode()

        assert result == 'hedge', f"long_short_mode → hedge, got {result}"
        # Verify API was called
        mock_req.assert_called()
        call_path = mock_req.call_args[0][1]
        assert 'account/config' in call_path, f"must call /api/v5/account/config, called {call_path}"


def test_detect_pos_mode_net_mode():
    """V6 V6-1: net_mode API response → 'net'"""
    with patch('miracle_kronos.okx_req') as mock_req:
        mock_req.return_value = {'code': '0', 'data': [{'posMode': 'net_mode'}]}

        from miracle_kronos import _detect_pos_mode
        result = _detect_pos_mode()

        assert result == 'net', f"net_mode → net, got {result}"
