"""Record the computing environment used for the analyses (no patient data are read).

Usage (from the project folder):
    python environment_report.py
Writes local_output/environment.json and prints it.
"""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def processor_name():
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor() or None


def memory_gb():
    if sys.platform == 'win32':
        try:
            out = subprocess.run(['powershell', '-NoProfile', '-Command',
                                  '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            return round(int(out) / 1024 ** 3, 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


report = {
    'python': platform.python_version(),
    'implementation': platform.python_implementation(),
    'os': platform.platform(),
    'processor': processor_name(),
    'logical_cpus': os.cpu_count(),
    'memory_gb': memory_gb(),
    'packages': {name: version(name) for name in ('numpy', 'pandas', 'matplotlib', 'scipy', 'python-docx')},
}
os.makedirs('local_output', exist_ok=True)
with open(os.path.join('local_output', 'environment.json'), 'w', encoding='utf-8') as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
