#!/usr/bin/env python3
"""Scenario-style tests for SpotExecutor using mocked Binance client calls."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from binance.error import ClientError

from spot_executor import BinanceCredentials, SpotExecutor, SpotExecutorError


class SpotExecutorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        creds = BinanceCredentials(api_key="key", api_secret="secret")
        self.executor = SpotExecutor(credentials=creds)
        self.mock_client = MagicMock()
        self.executor.client = self.mock_client

    def test_market_order_payload(self) -> None:
        expected = {"orderId": 1}
        self.mock_client.new_order.return_value = expected

        result = self.executor.place_market_order("btcusdt", "buy", "0.1")

        self.assertEqual(result, expected)
        self.mock_client.new_order.assert_called_once_with(
            symbol="BTCUSDT", side="BUY", type="MARKET", quantity="0.1"
        )

    def test_limit_order_payload(self) -> None:
        expected = {"orderId": 2}
        self.mock_client.new_order.return_value = expected

        result = self.executor.place_limit_order("ethusdt", "sell", "0.5", "2000", time_in_force="IOC")

        self.assertEqual(result, expected)
        self.mock_client.new_order.assert_called_once_with(
            symbol="ETHUSDT",
            side="SELL",
            type="LIMIT",
            timeInForce="IOC",
            quantity="0.5",
            price="2000",
        )

    def test_entry_with_protection_market(self) -> None:
        entry_response = {"orderId": 10}
        exit_response = {"orderListId": 55}
        self.mock_client.new_order.return_value = entry_response
        self.mock_client.oco_order.return_value = exit_response

        result = self.executor.place_entry_with_protection(
            symbol="bnbusdt",
            side="buy",
            entry_type="market",
            quantity="1",
            entry_price=None,
            take_profit_price="350",
            stop_price="280",
            stop_limit_price="275",
        )

        self.assertEqual(result["entry"], entry_response)
        self.assertEqual(result["exit"], exit_response)
        self.mock_client.new_order.assert_called_once_with(
            symbol="BNBUSDT", side="BUY", type="MARKET", quantity="1"
        )
        self.mock_client.oco_order.assert_called_once_with(
            symbol="BNBUSDT",
            side="SELL",
            quantity="1",
            price="350",
            stopPrice="280",
            stopLimitPrice="275",
            stopLimitTimeInForce="GTC",
        )

    def test_entry_with_protection_limit_requires_price(self) -> None:
        with self.assertRaises(SpotExecutorError):
            self.executor.place_entry_with_protection(
                symbol="bnbusdt",
                side="buy",
                entry_type="limit",
                quantity="1",
                entry_price=None,
                take_profit_price="350",
                stop_price="280",
            )

    def test_cancel_order_requires_identifier(self) -> None:
        with self.assertRaises(SpotExecutorError):
            self.executor.cancel_order("btcusdt")

    def test_client_error_wrapped(self) -> None:
        def boom(**_: str) -> None:
            raise ClientError(status_code=400, error_code=-1013, error_message="Bad", header={})

        self.executor.client.new_order.side_effect = boom

        with self.assertRaises(SpotExecutorError):
            self.executor.place_market_order("btcusdt", "buy", "0.1")

    def test_account_balances_hides_zero(self) -> None:
        self.mock_client.account.return_value = {
            "balances": [
                {"asset": "USDT", "free": "1000", "locked": "0"},
                {"asset": "BTC", "free": "0", "locked": "0"},
            ]
        }

        balances = self.executor.account_balances()

        self.assertIn("USDT", balances)
        self.assertNotIn("BTC", balances)


if __name__ == "__main__":
    unittest.main()
