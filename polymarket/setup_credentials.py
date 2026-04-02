"""
Utilitaire de configuration des credentials Polymarket.

Exécute ce script UNE SEULE FOIS pour régénérer les clés API CLOB.
Les nouvelles clés seront affichées — copie-les dans ton fichier .env.

Usage:
    python -m polymarket.setup_credentials
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    print("\n==========================================")
    print("  Polymarket — Génération des credentials")
    print("==========================================\n")

    from py_clob_client.client import ClobClient

    host     = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
    key: str      = os.getenv("POLYMARKET_PRIVATE_KEY") or ""
    funder: str   = os.getenv("POLYMARKET_FUNDER") or ""
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

    if not key:
        print("[FAIL] POLYMARKET_PRIVATE_KEY manquant dans .env")
        sys.exit(1)

    print(f"Host     : {host}")
    print(f"Chain ID : {chain_id}")
    print(f"Funder   : {funder}")
    print(f"Sig type : {sig_type}")
    print()

    # Niveau 1 seulement — pas besoin des creds pour les créer
    client = ClobClient(
        host=host,
        chain_id=chain_id,
        key=key,
        signature_type=sig_type,
        funder=funder,
    )

    print("Génération des credentials CLOB (Level 1 auth)...")
    try:
        creds = client.create_or_derive_api_creds()
        print("\n[OK] Credentials générés avec succès !\n")
        print("Copie ces valeurs dans ton fichier .env :")
        print("-" * 50)
        print(f"POLYMARKET_API_KEY={creds.api_key}")
        print(f"POLYMARKET_API_SECRET={creds.api_secret}")
        print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
        print("-" * 50)
        print()
        print("Puis relance le test : python -m polymarket.test_dry_run")
    except Exception as exc:
        print(f"\n[FAIL] Erreur : {exc}")
        print("\nVérifie que ta clé privée et ton adresse funder sont corrects.")
        sys.exit(1)


if __name__ == "__main__":
    main()
