import sys,subprocess,json,tempfile
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_mellum_humaneval import extract_completion

PROBLEM={'prompt':'def add(a,b):\n    """Sum."""\n','entry_point':'add'}
@pytest.mark.parametrize('raw,status',[
 ('<think>still thinking','unfinished_thinking'),
 ('<think>done</think>\n```python\ndef add(a,b):\n    return a+b\n```','extracted'),
 ('    return a+b','extracted'),('def add(:','invalid_python'),('','empty_answer')])
def test_extraction_fixed_without_tests(raw,status):assert extract_completion(raw,PROBLEM)[1]==status


def test_sandbox_blocks_workspace_write_and_network(tmp_path):
    # A fresh subprocess is essential: restrictions must not affect pytest itself.
    script=Path(__file__).resolve().parents[1]/'humaneval_sandbox.py'
    code='''import importlib.util,sys,os,socket,pathlib
spec=importlib.util.spec_from_file_location('sandbox',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
work=pathlib.Path(sys.argv[2]);os.chmod(work,0o777)
report=m.restrict(work,['/usr','/lib','/lib64','/dev/null'])
(work/'inside').write_text('ok')
for operation in [lambda:socket.socket(socket.AF_INET,socket.SOCK_STREAM),lambda:pathlib.Path('/workspace/forbidden-test').write_text('bad')]:
 try:operation()
 except PermissionError:pass
 else:raise AssertionError('Boundary did not hold')
print(report)
'''
    # pytest parents can be mode700; use a direct /tmp directory for uid65534.
    with tempfile.TemporaryDirectory(prefix='sandbox-test-') as work:
        p=subprocess.run([sys.executable,'-c',code,str(script),work],capture_output=True,text=True)
        if 'Landlock ABI' in p.stderr:
            pytest.skip('Managed tool process lacks Landlock; user-terminal sandbox self-test required')
        assert p.returncode==0,p.stderr
        assert "'uid': 0," not in p.stdout
