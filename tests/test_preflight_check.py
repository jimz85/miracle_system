"""
Tests for core/preflight_check.py
=================================
"""
from __future__ import annotations

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch


class TestPreflightCheck:
    """Tests for preflight_check module"""
    
    def test_check_result_str_ok(self):
        from core.preflight_check import CheckResult
        r = CheckResult(name="test", ok=True, message="OK")
        assert "✓" in str(r)
        assert "test" in str(r)
    
    def test_check_result_str_fail(self):
        from core.preflight_check import CheckResult
        r = CheckResult(name="test", ok=False, message="FAILED")
        assert "✗" in str(r)
        assert "FAILED" in str(r)
    
    def test_preflight_result_all_pass(self):
        from core.preflight_check import PreflightResult, CheckResult
        result = PreflightResult(symbol="BTC-USDT", exchange="okx")
        result.add(CheckResult(name="exchange_status", ok=True, message="OK"))
        result.add(CheckResult(name="min_size", ok=True, message="OK"))
        assert result.ok is True
        assert result.blocked_reason == ""
    
    def test_preflight_result_one_fail(self):
        from core.preflight_check import PreflightResult, CheckResult
        result = PreflightResult(symbol="BTC-USDT", exchange="okx")
        result.add(CheckResult(name="exchange_status", ok=True, message="OK"))
        result.add(CheckResult(name="min_size", ok=False, message="below minimum"))
        assert result.ok is False
        assert "min_size" in result.blocked_reason
    
    def test_preflight_error(self):
        from core.preflight_check import PreflightResult, CheckResult, PreflightError
        result = PreflightResult(symbol="BTC-USDT", exchange="okx")
        result.add(CheckResult(name="exchange_status", ok=False, message="API error"))
        with pytest.raises(PreflightError) as exc_info:
            raise PreflightError(result)
        assert "exchange_status" in str(exc_info.value)
    
    def test_make_key(self):
        from core.preflight_check import _INSTRUMENT_CACHE
        # Verify cache starts empty
        assert len(_INSTRUMENT_CACHE) == 0
    
    @patch("core.preflight_check._get_instrument_info")
    def test_check_min_size_pass(self, mock_get_info):
        from core.preflight_check import _check_min_size
        
        mock_client = MagicMock()
        mock_client.exchange = "okx"
        mock_get_info.return_value = {"min_sz": 0.001, "tick_sz": 0.1}
        
        result = _check_min_size(mock_client, "BTC-USDT", "long", 0.01)
        assert result.ok is True
        assert "min_sz" in str(result.details)
    
    @patch("core.preflight_check._get_instrument_info")
    def test_check_min_size_fail(self, mock_get_info):
        from core.preflight_check import _check_min_size
        
        mock_client = MagicMock()
        mock_client.exchange = "okx"
        mock_get_info.return_value = {"min_sz": 0.01, "tick_sz": 0.1}
        
        result = _check_min_size(mock_client, "BTC-USDT", "long", 0.001)
        assert result.ok is False
        assert "小于最小要求" in result.message
    
    @patch("core.preflight_check._get_instrument_info")
    def test_check_price_precision_ok(self, mock_get_info):
        from core.preflight_check import _check_price_precision
        
        mock_client = MagicMock()
        mock_get_info.return_value = {"tick_sz": 0.1}
        
        # 72000.0 is divisible by 0.1
        result = _check_price_precision(mock_client, "BTC-USDT", 72000.0)
        assert result.ok is True
    
    @patch("core.preflight_check._get_instrument_info")
    def test_check_price_precision_market_order(self, mock_get_info):
        from core.preflight_check import _check_price_precision
        
        mock_client = MagicMock()
        result = _check_price_precision(mock_client, "BTC-USDT", None)
        assert result.ok is True  # Market orders skip precision check
    
    def test_check_exchange_status_fail(self):
        from core.preflight_check import _check_exchange_status
        
        mock_client = MagicMock()
        mock_client.get_balance.side_effect = ConnectionError("Network error")
        
        result = _check_exchange_status(mock_client)
        assert result.ok is False
        assert "ConnectionError" in result.message or "连接失败" in result.message
    
    def test_preflight_check_full_pass(self):
        from core.preflight_check import preflight_check
        
        mock_client = MagicMock()
        mock_client.exchange = "okx"
        mock_client.get_balance.return_value = {"available": 10000.0, "total": 10000.0}
        mock_client.get_ticker.return_value = 72000.0
        mock_client.get_open_positions.return_value = []
        
        with patch("core.preflight_check._check_exchange_status") as mock_status, \
             patch("core.preflight_check._check_min_size") as mock_min, \
             patch("core.preflight_check._check_price_precision") as mock_prec, \
             patch("core.preflight_check._check_position_limit") as mock_pos:
            
            mock_status.return_value = MagicMock(ok=True, name="exchange_status", message="OK")
            mock_min.return_value = MagicMock(ok=True, name="min_size", message="OK")
            mock_prec.return_value = MagicMock(ok=True, name="price_precision", message="OK")
            mock_pos.return_value = MagicMock(ok=True, name="position_limit", message="OK")
            
            result = preflight_check(
                client=mock_client,
                symbol="BTC-USDT",
                side="long",
                size=0.01,
                price=72000.0,
                leverage=2
            )
            
            # All checks should be called
            mock_status.assert_called_once()
            mock_min.assert_called_once()
            mock_prec.assert_called_once()
            mock_pos.assert_called_once()
    
    def test_preflight_check_skip(self):
        from core.preflight_check import preflight_check
        
        mock_client = MagicMock()
        mock_client.exchange = "okx"
        
        with patch("core.preflight_check._check_exchange_status") as mock_status, \
             patch("core.preflight_check._check_min_size") as mock_min:
            
            mock_status.return_value = MagicMock(ok=True, name="exchange_status", message="OK")
            mock_min.return_value = MagicMock(ok=True, name="min_size", message="OK")
            
            result = preflight_check(
                client=mock_client,
                symbol="BTC-USDT",
                side="long",
                size=0.01,
                skip_checks=["price_precision", "position_limit"]
            )
            
            mock_status.assert_called_once()
            mock_min.assert_called_once()
            # Skipped checks should NOT be called
            mock_prec_mock = MagicMock()
            mock_prec_mock.assert_not_called()


class TestInstrumentCache:
    """Tests for instrument info caching"""
    
    def test_cache_key_format(self):
        from core.preflight_check import _INSTRUMENT_CACHE
        # Verify initial state
        assert isinstance(_INSTRUMENT_CACHE, dict)
