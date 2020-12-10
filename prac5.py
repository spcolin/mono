"""
test for the format of annotation.json
"""
import json

anno_path="/home/colin/anno/train_annotations.json"

js_file=open(anno_path)
js=json.load(js_file)

print(js[0])