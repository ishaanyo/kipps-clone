"""Generate LiveKit access tokens for browser participants."""
import os
from datetime import timedelta


def create_participant_token(
    identity: str,
    room_name: str,
    ttl_hours: int = 2,
) -> str:
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET required")

    try:
        from livekit.api import AccessToken, VideoGrants
        token = (
            AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
        )
        # ttl API varies by version
        if hasattr(token, "with_ttl"):
            token = token.with_ttl(timedelta(hours=ttl_hours))
        return token.to_jwt()
    except Exception:
        # Fallback older style
        from livekit import api
        grant = api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True)
        token = api.AccessToken(api_key, api_secret)
        token.with_identity(identity).with_name(identity).with_grants(grant)
        return token.to_jwt()
