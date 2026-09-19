"""Import and API conventions of the package, checked so that a change cannot break them unnoticed."""

import ast
import pathlib
import re

import mantispy as mt

SRC = pathlib.Path(mt.__file__).parent

#: Libraries mantispy is checked against rather than built on. They are test-time
#: dependencies: importing one from src would make it a runtime dependency of a package
#: that reimplements what it does, and would make the equivalence tests circular.
#: harmonypy is absent because pp.harmony imports it as an optional runtime dependency.
REFERENCE_LIBRARIES = {
    "pycytominer",
    "pyod",
    "scib",
    "scib_metrics",
    "cytominer_eval",
    "scmorph",
}

#: Functions scanpy provides and users call directly. A thin wrapper would add a name to
#: learn, a signature to keep in sync, and a place for defaults to drift.
SCANPY_TERRITORY = {"pca", "neighbors", "umap", "tsne", "leiden", "louvain", "draw_graph", "harmony_integrate"}


def _imports(path: pathlib.Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_src_never_imports_a_reference_library():
    offending = {
        str(path.relative_to(SRC)): sorted(found)
        for path in sorted(SRC.rglob("*.py"))
        if (found := _imports(path) & REFERENCE_LIBRARIES)
    }
    assert not offending, offending


def test_only_core_reads_the_matrix_directly():
    """Outside _core every read of X or of a layer goes through get_matrix, so a streaming
    backend has one place to change."""
    offending = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.parent.name == "_core":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr in {"X", "layers"}):
                continue
            # Writes are allowed; a function has to store its result.
            if isinstance(getattr(node, "ctx", None), ast.Store):
                continue
            source = ast.unparse(node)
            if source.endswith((".X", ".layers")) and "self" not in source:
                offending.setdefault(str(path.relative_to(SRC)), set()).add(source)

    # Writes through subscripts (adata.layers[key] = ...) and membership tests are allowed;
    # reads of the values are not.
    reads = {
        path: sorted(names) for path, names in offending.items() if any(not name.endswith(".layers") for name in names)
    }
    assert not reads, reads


def test_no_wrappers_for_what_scanpy_already_does():
    for namespace in (mt.pp, mt.tl, mt.pl, mt.get, mt.metrics):
        clashing = {name for name in dir(namespace) if name in SCANPY_TERRITORY}
        assert not clashing, f"mt.{namespace.__name__.split('.')[-1]} defines {sorted(clashing)}"


def test_every_cited_key_is_in_the_bibliography():
    """Sphinx checks the keys it renders; this also covers comments, private modules and tests."""
    root = pathlib.Path(__file__).parents[1]
    known = set(re.findall(r"^@\w+\{(\w+),", (root / "docs/references.bib").read_text(), re.MULTILINE))
    paths = [
        *SRC.rglob("*.py"),
        *root.glob("tests/*.py"),
        *root.glob("docs/*.md"),
        *root.glob("docs/*/*.md"),
        *root.glob("docs/*/*.ipynb"),
    ]
    cited = {
        key.strip()
        for path in paths
        for keys in re.findall(r"(?::cite:[tp]:|\{cite:[tp]\})`(\w+(?:,\s*\w+)*)`", path.read_text())
        for key in keys.split(",")
    }
    assert cited, "no citations found"
    assert cited <= known, sorted(cited - known)
