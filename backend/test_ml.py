from app.services import ml_service
import json, os
p = "/tmp/test_food.jpg"
# create a small placeholder file
open(p, "wb").close()
r = ml_service.infer_image(p)
print(json.dumps(r, indent=2))
