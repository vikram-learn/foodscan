# backend/app/api/v1/__init__.py
# Import v1 routers here so app.main can include them.
# Add any new routers to this list (user, auth, scan, quantum, etc.)

from . import user
from . import auth
from . import scan
# if you have quantum router, keep it too:
try:
    from . import quantum
except Exception:
    # quantum may be optional in dev if service not created yet
    pass
