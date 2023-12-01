from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'
# Webcam
WEBCAM_PATH = 0

# N stick
# H battery
# H chemical_spray_can
# N plastic_box
# R plastic_bottle
# H chemical_plastic_bottle
# N cardboard_bowl
# N straw
# R plastic_bottle_cap
# N plastic_cup
# N scrap_plastic
# N plastic_bag
# N scrap_paper
# R can
# N snack_bag
# H light_bulb
# N plastic_spoon
# H chemical_plastic_gallon
# N plastic_cup_lid
# H paint_bucket
# R cardboard_box

# types of waste
RECYCLABLE = ['plastic_bottle','plastic_cup','scrap_plastic','plastic_bag','scrap_paper','snack_bag','plastic_spoon','plastic_cup_lid']
NON_RECYCLABLE = ['stick','plastic_box','cardboard_bowl','straw','plastic_bottle_cap','can','cardboard_box']
HAZARDOUS = ['battery', 'chemical_spray_can','chemical_plastic_bottle','light_bulb','chemical_plastic_gallon','paint_bucket']