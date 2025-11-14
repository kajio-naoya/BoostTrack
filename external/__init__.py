import sys
import os
# Add external/deep-person-reid to the beginning of sys.path to prioritize it over installed torchreid
deep_person_reid_path = os.path.join(os.getcwd(), "external", "deep-person-reid")
if os.path.exists(deep_person_reid_path):
    sys.path.insert(0, deep_person_reid_path)
sys.path.append(os.path.join(os.getcwd(), "external"))
