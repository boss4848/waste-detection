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

# R cardboard_box
# R can
# N plastic_bag
# N scrap_paper
# N stick
# N plastic_cup
# N snack_bag
# R plastic_bottle_cap
# N plastic_box
# H battery
# N straw
# H chemical_spray_can
# N plastic_cup_lid
# R plastic_bottle
# R reuseable_paper
# N scrap_plastic
# H chemical_plastic_bottle
# H chemical_plastic_gallon
# N cardboard_bowl
# H light_bulb
# N plastic_cultery
# H paint_bucket
# types of waste
RECYCLABLE = ['cardboard_box','can','plastic_bottle_cap','plastic_bottle','reuseable_paper']
NON_RECYCLABLE = ['plastic_bag','scrap_paper','stick','plastic_cup','snack_bag','plastic_box','straw','plastic_cup_lid','scrap_plastic','cardboard_bowl','plastic_cultery']
HAZARDOUS = ['battery','chemical_spray_can','chemical_plastic_bottle','chemical_plastic_gallon','light_bulb','paint_bucket']