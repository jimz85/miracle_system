"""
Tests for Funding Rate & OI factors in price_factors.py
Run with: pytest tests/test_funding_rate_factor.py -v
"""

import pytest
from core.price_factors import PriceFactors


class TestCalcFundingRateFactor:
    """Tests for calc_funding_rate_factor()"""

    def test_empty_input(self):
        """空输入返回默认值"""
        result = PriceFactors.calc_funding_rate_factor([])
        assert result["funding_rate"] == 0.0
        assert result["funding_rate_trend"] == 0.0
        assert result["funding_rate_direction"] == "stable"
        assert result["is_high_funding"] is False

    def test_basic_funding_rate(self):
        """基本资金费率计算"""
        funding_rates = [0.0001, 0.0002, 0.0003]
        result = PriceFactors.calc_funding_rate_factor(funding_rates)
        assert result["funding_rate"] == 0.0003
        assert result["funding_rate_trend"] == pytest.approx(0.0002, rel=1e-9)
        assert result["is_high_funding"] is False  # 0.0003 不超过 0.0003 threshold

    def test_high_funding_short_boost(self):
        """做空 + 高资金费率 = 信心加成"""
        funding_rates = [0.0004, 0.0004, 0.0005]
        result = PriceFactors.calc_funding_rate_factor(funding_rates, side="short")
        assert result["is_high_funding"] is True
        assert result["short_high_funding_boost"] > 0
        assert result["confidence_boost"] > 0

    def test_high_funding_long_penalty(self):
        """做多 + 高资金费率 = 轻微惩罚"""
        funding_rates = [0.0004, 0.0004, 0.0005]
        result = PriceFactors.calc_funding_rate_factor(funding_rates, side="long")
        assert result["is_high_funding"] is True
        assert result["short_high_funding_boost"] == -0.05  # 轻微惩罚

    def test_funding_rate_increasing(self):
        """资金费率上升趋势"""
        funding_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        result = PriceFactors.calc_funding_rate_factor(funding_rates, side="short")
        assert result["funding_rate_direction"] == "increasing"

    def test_funding_rate_decreasing(self):
        """资金费率下降趋势"""
        funding_rates = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
        result = PriceFactors.calc_funding_rate_factor(funding_rates)
        assert result["funding_rate_direction"] == "decreasing"

    def test_short_with_increasing_funding(self):
        """做空 + 资金费率上升 = 额外加成"""
        funding_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        result = PriceFactors.calc_funding_rate_factor(funding_rates, side="short")
        assert result["funding_rate_direction"] == "increasing"
        # increasing + short + high funding 方向加成已包含在 short_high_funding_boost 中
        assert result["confidence_boost"] >= result["short_high_funding_boost"]

    def test_funding_rate_trend_calculation(self):
        """3期均值计算正确"""
        funding_rates = [0.0001, 0.0003, 0.0002]  # trend = (0.0003+0.0002+0.0001)/3
        result = PriceFactors.calc_funding_rate_factor(funding_rates)
        expected_trend = (0.0001 + 0.0003 + 0.0002) / 3
        assert result["funding_rate_trend"] == pytest.approx(expected_trend, rel=1e-9)


class TestCalcOiChangeRate:
    """Tests for calc_oi_change_rate()"""

    def test_empty_input(self):
        """空输入返回默认值"""
        result = PriceFactors.calc_oi_change_rate([])
        assert result["oi_change_rate"] == 0.0
        assert result["oi_direction"] == "stable"
        assert result["is_filtered"] is False

    def test_single_value(self):
        """单值返回默认"""
        result = PriceFactors.calc_oi_change_rate([100.0])
        assert result["oi_current"] == 100.0
        assert result["oi_change_rate"] == 0.0

    def test_oi_increasing(self):
        """OI上升"""
        oi_history = [100.0, 110.0, 120.0]  # +20%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["oi_direction"] == "increasing"
        assert result["oi_change_rate"] == pytest.approx(0.20, rel=1e-6)

    def test_oi_decreasing(self):
        """OI下降"""
        oi_history = [100.0, 90.0, 80.0]  # -20%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["oi_direction"] == "decreasing"
        assert result["oi_change_rate"] == pytest.approx(-0.20, rel=1e-6)

    def test_oi_stable(self):
        """OI稳定"""
        oi_history = [100.0, 101.0, 102.0]  # +2%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["oi_direction"] == "stable"

    def test_oi_sharp_decline_filter(self):
        """OI急剧下降触发过滤"""
        oi_history = [100.0, 70.0, 50.0]  # -50%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["is_filtered"] is True
        assert result["filter_reason"] == "oi_sharp_decline"
        assert result["confidence_penalty"] == 0.15

    def test_oi_moderate_decline_penalty(self):
        """OI温和下降轻微惩罚"""
        oi_history = [100.0, 85.0, 80.0]  # -15%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["is_filtered"] is False
        assert result["filter_reason"] == "oi_moderate_decline"
        assert result["confidence_penalty"] == 0.05

    def test_oi_increasing_no_filter(self):
        """OI上升无过滤"""
        oi_history = [100.0, 120.0, 150.0]  # +50%
        result = PriceFactors.calc_oi_change_rate(oi_history)
        assert result["is_filtered"] is False
        assert result["confidence_penalty"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
