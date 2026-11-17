"""Execute and save the audit notebook with runtime files inside the project."""
import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter' / 'runtime')
os.environ['IPYTHONDIR'] = str(root / '.ipython')
import nbformat
from nbclient import NotebookClient
path = root / 'notebooks' / '01_data_audit.ipynb'
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code'
               for o in c.outputs)
print('Executed all audit cells successfully and saved outputs.')
