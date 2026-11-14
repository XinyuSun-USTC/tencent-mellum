"""Linux CPU evaluation isolation: Landlock filesystem, dropped uid, seccomp network.

Applied in a disposable scoring process before official evaluator starts workers.
No fallback to unrestricted execution if kernel features are unavailable.
"""
import ctypes,errno,os,platform,resource
from pathlib import Path


def _landlock(work, read_paths):
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise RuntimeError('Sandbox requires Linux x86_64')
    libc=ctypes.CDLL(None,use_errno=True)
    def check(value,label):
        if value<0:raise OSError(ctypes.get_errno(),label)
        return value
    check(libc.prctl(38,1,0,0,0),'no_new_privs')
    abi=check(libc.syscall(444,0,0,1),'Landlock ABI')
    handled=(1<<13)-1
    if abi>=2:handled|=1<<13
    if abi>=3:handled|=1<<14
    class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
    class Rule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    attr=Ruleset(handled);fd=check(libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0),'Landlock create')
    try:
        rules=[(Path(p).resolve(),(1<<2)|(1<<3)) for p in read_paths if Path(p).exists()]
        rules += [(Path(work).resolve(),handled)]
        for path,access in rules:
            if path.is_file():access &= ~(1<<3)
            pathfd=os.open(path,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=Rule(access,pathfd);check(libc.syscall(445,fd,1,ctypes.byref(rule),0),'Landlock rule')
            finally:os.close(pathfd)
        check(libc.syscall(446,fd,0),'Landlock restrict')
    finally:os.close(fd)
    return {'landlock_abi':abi,'filesystem':'Landlock: write only private workdir'}


def restrict(work, read_paths, chroot_root=None):
    try:
        filesystem=_landlock(work,read_paths)
    except OSError:
        if chroot_root is None:raise
        # Read-only runtime copies owned by root; only work is owned by nobody.
        os.chroot(chroot_root);os.chdir(str(work))
        filesystem={'filesystem':'chroot runtime copy; uid cannot write runtime'}
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0)<0:raise OSError(ctypes.get_errno(),'no_new_privs')
    def check(value,label):
        if value<0:raise OSError(ctypes.get_errno(),label)
        return value
    class Insn(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Insn))]
    deny=0x50000|errno.EPERM;allow=0x7fff0000
    # Validate native syscall architecture; kill x32 ABI rather than bypass rules.
    code=[Insn(0x20,0,0,4),Insn(0x15,1,0,0xc000003e),Insn(6,0,0,0x80000000),
          Insn(0x20,0,0,0),Insn(0x35,0,1,0x40000000),Insn(6,0,0,deny)]
    for nr in [101,165,166,272,298,308,310,311,321,248,250,249,175,176,246]:
        code += [Insn(0x15,0,1,nr),Insn(6,0,0,deny)]
    # Only AF_UNIX sockets (official multiprocessing Manager IPC) are allowed.
    code += [Insn(0x15,0,3,41),Insn(0x20,0,0,16),Insn(0x15,1,0,1),Insn(6,0,0,deny),Insn(6,0,0,allow)]
    array=(Insn*len(code))(*code);program=Program(len(code),array)
    check(libc.prctl(22,2,ctypes.byref(program),0,0),'seccomp')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
    if os.geteuid()==0:
        os.setgroups([]);os.setgid(65534);os.setuid(65534)
    if os.geteuid()==0:raise RuntimeError('Refuse evaluation as root')
    return {**filesystem,'uid':os.geteuid(),'network':'AF_UNIX only'}
