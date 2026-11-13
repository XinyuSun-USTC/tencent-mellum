"""Export semantic fields with an explicitly chosen isolated interpreter."""
import argparse
import importlib.util
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['atom','hf'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
if a.backend=='atom':
    spec=importlib.util.spec_from_file_location('mellum_config_only','/workspace/ATOM/atom/model_config/mellum.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.MellumConfig.from_pretrained(model,local_files_only=True)
else:
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(model,local_files_only=True)
raw=json.loads((model/'config.json').read_text())
fields={k:getattr(config,k) for k in raw if k not in ['transformers_version','dtype']}
fields['dtype']=str(config.dtype)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(fields,sort_keys=True,indent=2)+'\n')
