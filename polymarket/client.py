"""
Initialise le ClobClient Polymarket depuis les variables d'environnement.
"""
import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

load_dotenv()


def build_client() -> ClobClient:
    """Retourne un ClobClient Level 2 (lecture + écriture d'ordres)."""
    host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
    private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY") or ""
    funder: str = os.getenv("POLYMARKET_FUNDER") or ""
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

    creds = ApiCreds(
        api_key=os.getenv("POLYMARKET_API_KEY") or "",
        api_secret=os.getenv("POLYMARKET_API_SECRET") or "",
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE") or "",
    )

    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=private_key,
        creds=creds,
        signature_type=sig_type,
        funder=funder,
    )
