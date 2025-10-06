#!/usr/bin/env python3
"""High-level helper for driving Binance Spot Testnet trading flows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from binance.error import ClientError
from binance.spot import Spot as SpotClient
from dotenv import load_dotenv

TESTNET_BASE_URL = "https://testnet.binance.vision"


class SpotExecutorError(RuntimeError):
    """Raised when a Binance API call fails."""


@dataclass
class BinanceCredentials:
    api_key: str
    api_secret: str


def load_credentials(env_prefix: str = "BINANCE_TESTNET") -> BinanceCredentials:
    load_dotenv()
    api_key = os.getenv(f"{env_prefix}_API_KEY")
    api_secret = os.getenv(f"{env_prefix}_API_SECRET")
    if not api_key or not api_secret:
        raise SpotExecutorError(
            f"Missing credentials: set {env_prefix}_API_KEY and {env_prefix}_API_SECRET in the environment or .env"
        )
    return BinanceCredentials(api_key=api_key, api_secret=api_secret)


class SpotExecutor:
    def __init__(
        self,
        credentials: Optional[BinanceCredentials] = None,
        base_url: str = TESTNET_BASE_URL,
        env_prefix: str = "BINANCE_TESTNET",
    ) -> None:
        if credentials is None:
            credentials = load_credentials(env_prefix)
        self.client = SpotClient(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            base_url=base_url,
        )

    def ping(self) -> Dict[str, Any]:
        return self._call(self.client.ping)

    def server_time(self) -> Dict[str, Any]:
        return self._call(self.client.time)

    def account_balances(self, hide_zero: bool = True) -> Dict[str, Dict[str, str]]:
        account = self._call(self.client.account)
        balances = account.get("balances", [])
        if hide_zero:
            balances = [b for b in balances if float(b.get("free", 0)) or float(b.get("locked", 0))]
        return {b["asset"]: {"free": b["free"], "locked": b["locked"]} for b in balances}

    def price_ticker(self, symbol: str) -> Dict[str, Any]:
        return self._call(self.client.ticker_price, symbol=symbol.upper())

    def place_market_order(self, symbol: str, side: str, quantity: str, **extra: Any) -> Dict[str, Any]:
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }
        payload.update(extra)
        return self._call(self.client.new_order, **payload)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        time_in_force: str = "GTC",
        **extra: Any,
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": quantity,
            "price": price,
        }
        payload.update(extra)
        return self._call(self.client.new_order, **payload)

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        take_profit_price: str,
        stop_price: str,
        stop_limit_price: Optional[str] = None,
        stop_limit_time_in_force: str = "GTC",
        **extra: Any,
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "price": take_profit_price,
            "stopPrice": stop_price,
            "stopLimitPrice": stop_limit_price or stop_price,
            "stopLimitTimeInForce": stop_limit_time_in_force,
        }
        payload.update(extra)
        return self._call(self.client.oco_order, **payload)

    def place_entry_with_protection(
        self,
        symbol: str,
        side: str,
        entry_type: str,
        quantity: str,
        entry_price: Optional[str],
        take_profit_price: str,
        stop_price: str,
        stop_limit_price: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        side = side.upper()
        entry_type = entry_type.upper()
        entry_response: Dict[str, Any]
        if entry_type == "MARKET":
            entry_response = self.place_market_order(symbol, side, quantity, **extra)
        elif entry_type == "LIMIT":
            if entry_price is None:
                raise SpotExecutorError("entry_price is required for LIMIT orders")
            entry_response = self.place_limit_order(
                symbol,
                side,
                quantity,
                entry_price,
                **extra,
            )
        else:
            raise SpotExecutorError(f"Unsupported entry_type: {entry_type}")

        exit_side = "SELL" if side == "BUY" else "BUY"
        exit_response = self.place_oco_order(
            symbol,
            exit_side,
            quantity,
            take_profit_price=take_profit_price,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
        )
        return {"entry": entry_response, "exit": exit_response}

    def sell_market(self, symbol: str, quantity: str, **extra: Any) -> Dict[str, Any]:
        return self.place_market_order(symbol, "SELL", quantity, **extra)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise SpotExecutorError("order_id or orig_client_order_id must be provided")
        payload: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            payload["orderId"] = order_id
        if orig_client_order_id is not None:
            payload["origClientOrderId"] = orig_client_order_id
        return self._call(self.client.cancel_order, **payload)

    def cancel_open_orders(self, symbol: str) -> Dict[str, Any]:
        return self._call(self.client.cancel_open_orders, symbol=symbol.upper())

    def get_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        if order_id is None and orig_client_order_id is None:
            raise SpotExecutorError("order_id or orig_client_order_id must be provided")
        payload: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            payload["orderId"] = order_id
        if orig_client_order_id is not None:
            payload["origClientOrderId"] = orig_client_order_id
        return self._call(self.client.get_order, **payload)

    def open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        if symbol:
            return self._call(self.client.get_open_orders, symbol=symbol.upper())
        return self._call(self.client.get_open_orders)

    def trade_history(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        return self._call(self.client.my_trades, symbol=symbol.upper(), limit=limit)

    def _call(self, func, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ClientError as exc:
            raise SpotExecutorError(f"Binance API error ({exc.error_code}): {exc.error_message}") from exc


def demo() -> None:
    executor = SpotExecutor()
    print("Server time:", executor.server_time())
    print("Account balances:", executor.account_balances())


if __name__ == "__main__":
    demo()
