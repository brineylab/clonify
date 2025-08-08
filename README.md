clonify (native)

Install
- pip: `pip install .[dev]`
- uv: `uv pip install .[dev]`

Usage
```python
from clonify import clonify
assign_dict, df_out = clonify(df, distance_cutoff=0.35)
```

Develop
- Build dev: `pip install maturin && maturin develop`
- Run tests: `pytest`


