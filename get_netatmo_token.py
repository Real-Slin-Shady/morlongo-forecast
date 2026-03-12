#!/usr/bin/env python3
"""
One-time script to get Netatmo refresh_token.
Run this locally, then add the token to GitHub Secrets.

Usage:
    python get_netatmo_token.py

You'll need:
    - client_id and client_secret from https://dev.netatmo.com/apps
"""

import webbrowser
import http.server
import urllib.parse
import requests
import json

# Netatmo OAuth endpoints
AUTH_URL = "https://api.netatmo.com/oauth2/authorize"
TOKEN_URL = "https://api.netatmo.com/oauth2/token"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"

# Scopes needed for weather station
SCOPES = "read_station"


def main():
    print("=" * 60)
    print("NETATMO TOKEN SETUP")
    print("=" * 60)
    print("\nThis script will help you get a refresh_token for the Netatmo API.")
    print("You'll need your client_id and client_secret from:")
    print("  https://dev.netatmo.com/apps\n")

    client_id = input("Enter your client_id: ").strip()
    client_secret = input("Enter your client_secret: ").strip()

    if not client_id or not client_secret:
        print("Error: Both client_id and client_secret are required.")
        return

    # Build authorization URL
    auth_params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": "morlongo"
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    print("\n" + "=" * 60)
    print("STEP 1: Authorize in browser")
    print("=" * 60)
    print(f"\nOpening browser to authorize...\n")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")

    # Start local server to catch the callback
    authorization_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal authorization_code
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            if "code" in params:
                authorization_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Success!</h1>
                    <p>Authorization code received. You can close this window.</p>
                    </body></html>
                """)
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Error: No code received</h1></body></html>")

        def log_message(self, format, *args):
            pass  # Suppress logging

    # Open browser
    webbrowser.open(auth_url)

    # Wait for callback
    print("Waiting for authorization...")
    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), CallbackHandler)
    server.handle_request()

    if not authorization_code:
        print("Error: No authorization code received.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: Exchange code for tokens")
    print("=" * 60)

    # Exchange authorization code for tokens
    token_data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": authorization_code,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES
    }

    response = requests.post(TOKEN_URL, data=token_data)

    if response.status_code != 200:
        print(f"Error getting tokens: {response.status_code}")
        print(response.text)
        return

    tokens = response.json()
    refresh_token = tokens.get("refresh_token")
    access_token = tokens.get("access_token")

    if not refresh_token:
        print("Error: No refresh_token in response")
        print(json.dumps(tokens, indent=2))
        return

    # Test the token by fetching station data
    print("\nTesting token by fetching your stations...")
    headers = {"Authorization": f"Bearer {access_token}"}
    stations_response = requests.get(
        "https://api.netatmo.com/api/getstationsdata",
        headers=headers
    )

    if stations_response.status_code == 200:
        data = stations_response.json()
        devices = data.get("body", {}).get("devices", [])
        print(f"\nFound {len(devices)} device(s):")
        for device in devices:
            print(f"  - {device.get('station_name', 'Unknown')} ({device.get('_id')})")
            for module in device.get("modules", []):
                print(f"    - {module.get('module_name', 'Unknown')} ({module.get('type')})")

    print("\n" + "=" * 60)
    print("SUCCESS! Here are your credentials:")
    print("=" * 60)
    print(f"\nNETATMO_CLIENT_ID:      {client_id}")
    print(f"NETATMO_CLIENT_SECRET:  {client_secret}")
    print(f"NETATMO_REFRESH_TOKEN:  {refresh_token}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
1. Go to: github.com/Real-Slin-Shady/morlongo-forecast/settings/secrets/actions

2. Add these 3 secrets:
   - NETATMO_CLIENT_ID     → paste your client_id
   - NETATMO_CLIENT_SECRET → paste your client_secret
   - NETATMO_REFRESH_TOKEN → paste the refresh_token above

3. Done! The workflow will now fetch real observations hourly.
""")


if __name__ == "__main__":
    main()
