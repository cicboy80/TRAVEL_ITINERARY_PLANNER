# travel_planner/ date_utils

from datetime import date, timedelta

def expand_dates(start_date: str, end_date: str):
    """Return a list of dates between start and end inclusive."""
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)
    days = (ed - sd).days + 1
    date_list = [(sd + timedelta(days=i)).isoformat() for i in range(days)]
    return days, date_list