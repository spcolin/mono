"""
test for the format of annotation.json
"""
import json

anno_path="/home/colin/anno/train_anno.json"

js_file=open(anno_path)
js=json.load(js_file)

print(js)

js_file.close()