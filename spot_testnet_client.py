from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from binance.error import ClientError
from binance.spot import Spot as SpotClient
from dotenv import load_dotenv

TESTNET_BASE_URL = "https://testnet.binance.vision"
SYMBOL = "BTCUSDT"
ORDER_SIDE = "BUY"
ORDER_TYPE = "LIMIT"
ORDER_QUANTITY = "0.001"
ORDER_PRICE = "109900.0"
ORDER_TIME_IN_FORCE = "GTC"


@dataclass
class BinanceCredentials:
    api_key: str
    api_secret: str


def load_credentials() -> BinanceCredentials:
    load_dotenv()
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in environment or .env")
    return BinanceCredentials(api_key=api_key, api_secret=api_secret)


def build_client(creds: BinanceCredentials) -> SpotClient:
    return SpotClient(api_key=creds.api_key, api_secret=creds.api_secret, base_url=TESTNET_BASE_URL)


def sanity_checks(client: SpotClient) -> Dict[str, Any]:
    return {
        "ping": client.ping(),
        "time": client.time(),
        "exchange_info": client.exchange_info(symbol=SYMBOL),
    }


def account_balances(client: SpotClient) -> Dict[str, Any]:
    account = client.account()
    balances = [b for b in account.get("balances", []) if float(b.get("free", 0)) > 0]
    return {b["asset"]: {"free": b["free"], "locked": b["locked"]} for b in balances}


def place_limit_order(client: SpotClient) -> Dict[str, Any]:
    return client.new_order(
        symbol=SYMBOL,
        side=ORDER_SIDE,
        type=ORDER_TYPE,
        timeInForce='GTC',
        quantity=ORDER_QUANTITY,
        price=ORDER_PRICE,
    )


def main() -> None:
    creds = load_credentials()
    client = build_client(creds)

    try:
        checks = sanity_checks(client)
        print(f"Ping: {checks['ping']}")
        print(f"Server time: {checks['time']}")
        symbol_info = checks['exchange_info']['symbols'][0]
        print(f"Symbol {symbol_info['symbol']} status: {symbol_info['status']}")

        balances = account_balances(client)
        if balances:
            print("Account balances:")
            for asset, info in balances.items():
                print(f"  {asset}: free={info['free']} locked={info['locked']}")
        else:
            print("Account balances: none (top up testnet funds if needed)")

        order_response = place_limit_order(client)
        print("Created order:")
        for key, value in order_response.items():
            print(f"  {key}: {value}")
    except ClientError as exc:
        print(f"Binance API error ({exc.error_code}): {exc.error_message}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
