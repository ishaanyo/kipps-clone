"""Generate LiveKit access tokens for browser participants."""
import os
from datetime import timedelta


def _env(name: str) -> str:
    val = (os.getenv(name) or "").strip().strip('"').strip("'")
    return val


def create_participant_token(
    identity: str,
    room_name: str,
    ttl_hours: int = 2,
) -> str:
    """
    Official pattern:
      from livekit import api
      api.AccessToken(key, secret).with_identity(...).with_grants(...).to_jwt()
    """
    api_key = _env("LIVEKIT_API_KEY")
    api_secret = _env("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET required")

    # Prefer livekit.api (livekit-api package)
    try:
        from livekit import api

        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .with_ttl(timedelta(hours=ttl_hours))
            .to_jwt()
        )
        return token
    except Exception as e1:
        try:
            from livekit.api import AccessToken, VideoGrants

            token = (
                AccessToken(api_key=api_key, api_secret=api_secret)
                .with_identity(identity)
                .with_name(identity)
                .with_grants(
                    VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_subscribe=True,
                        can_publish_data=True,
                    )
                )
                .with_ttl(timedelta(hours=ttl_hours))
                .to_jwt()
            )
            return token
        except Exception as e2:
            raise RuntimeError(f"Token generation failed: {e1} | {e2}") from e2
