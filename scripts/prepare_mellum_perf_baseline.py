"""CPU-only preparation of assessment A/B/C and the current shared-routing server."""
from datetime import datetime,timezone
from pathlib import Path
import json
from mellum_perf_protocol import prepare
from mellum_runtime_config import MODEL


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    out=Path('/workspace/results/mellum_perf_protocol')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    manifest=prepare(out,tokenizer)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('RESULT perf_protocol_prepared output',out)
if __name__=='__main__':main()
