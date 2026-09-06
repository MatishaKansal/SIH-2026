"""Minimal Supabase REST client for the application API."""

import asyncio
from typing import Any

import httpx

from app.config import settings


class SupabaseClient:
    def __init__(self) -> None:
        if not settings.NEXT_PUBLIC_SUPABASE_URL or not settings.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY:
            raise RuntimeError("Supabase URL and publishable key are required")

        self.base_url = settings.NEXT_PUBLIC_SUPABASE_URL.rstrip("/") + "/rest/v1"
        self.headers = {
            "apikey": settings.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY,
            "Authorization": f"Bearer {settings.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY}",
        }

    @property
    def server_headers(self) -> dict[str, str]:
        key = settings.SUPABASE_SERVICE_ROLE_KEY
        if not key:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required to write Supabase tables")
        return {"apikey": key, "Authorization": f"Bearer {key}"}

    async def select(
        self,
        table: str,
        columns: str = "*",
        params: dict[str, Any] | None = None,
        use_service_role: bool = False,
    ) -> list[dict[str, Any]]:
        query = {"select": columns, **(params or {})}
        timeout = httpx.Timeout(settings.AI_SERVICE_TIMEOUT, connect=5)
        headers = self.server_headers if use_service_role else self.headers
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"{self.base_url}/{table}", headers=headers, params=query)
                if response.status_code == 401:
                    raise RuntimeError(
                        "Supabase rejected the API key. Check the project URL and use this project's anon or valid publishable key."
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectTimeout, httpx.ReadTimeout):
                if attempt == 2:
                    raise
                await asyncio.sleep(0.25 * (attempt + 1))

    async def insert(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {**self.server_headers, "Content-Profile": "public", "Prefer": "return=representation"}
        async with httpx.AsyncClient(timeout=settings.AI_SERVICE_TIMEOUT) as client:
            response = await client.post(f"{self.base_url}/{table}", headers=headers, json=payload)
        if response.status_code == 401:
            raise RuntimeError("Supabase rejected the API key. Check the project URL and API key.")
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else {}

    async def update(self, table: str, filters: dict[str, str], payload: dict[str, Any]) -> None:
        headers = {**self.server_headers, "Content-Profile": "public", "Prefer": "return=minimal"}
        async with httpx.AsyncClient(timeout=settings.AI_SERVICE_TIMEOUT) as client:
            response = await client.patch(
                f"{self.base_url}/{table}",
                headers=headers,
                params=filters,
                json=payload,
            )
        if response.status_code == 401:
            raise RuntimeError("Supabase rejected the API key. Check the project URL and API key.")
        response.raise_for_status()

    async def auth_signup(self, email: str, password: str, full_name: str) -> dict[str, Any]:
        headers = {**self.server_headers, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=settings.AI_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{settings.NEXT_PUBLIC_SUPABASE_URL.rstrip('/')}/auth/v1/admin/users",
                headers=headers,
                json={
                    "email": email,
                    "password": password,
                    "email_confirm": True,
                    "user_metadata": {"full_name": full_name},
                },
            )
        response.raise_for_status()
        return response.json()

    async def auth_signin(self, email: str, password: str) -> dict[str, Any]:
        headers = {**self.headers, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=settings.AI_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{settings.NEXT_PUBLIC_SUPABASE_URL.rstrip('/')}/auth/v1/token?grant_type=password",
                headers=headers,
                json={"email": email, "password": password},
            )
        response.raise_for_status()
        return response.json()


supabase = SupabaseClient()