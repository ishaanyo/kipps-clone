from datetime import datetime, timedelta
from loguru import logger


# Simple in-memory calendar for demo. Replace with Google Calendar / Cal.com API.
_bookings = []


def book_appointment(
    name: str,
    phone: str,
    datetime_str: str,
    purpose: str = "Demo call",
) -> str:
    """
    Book an appointment.
    
    Args:
        name: Caller's full name
        phone: Caller's phone number
        datetime_str: Preferred date & time (e.g. "2026-09-17 15:00")
        purpose: Reason for the meeting
    """
    try:
        # Very basic validation
        dt = datetime.fromisoformat(datetime_str.replace(" ", "T"))
        booking = {
            "id": len(_bookings) + 1,
            "name": name,
            "phone": phone,
            "datetime": dt.isoformat(),
            "purpose": purpose,
            "created_at": datetime.utcnow().isoformat(),
        }
        _bookings.append(booking)
        logger.success(f"Appointment booked: {booking}")
        return (
            f"Successfully booked appointment for {name} on {dt.strftime('%A, %d %B %Y at %I:%M %p')}. "
            f"A confirmation will be sent to {phone}."
        )
    except Exception as e:
        return f"Could not book appointment: {str(e)}. Please provide a valid date and time like '2026-09-17 15:00'."


def get_available_slots(date: str = None) -> str:
    """Return mock available slots."""
    base = datetime.utcnow() + timedelta(days=1)
    slots = [
        (base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
        for i in [10, 11, 14, 15, 16]
    ]
    return "Available slots: " + ", ".join(slots)
