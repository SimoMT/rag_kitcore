from vectorstore.qdrant_store import InMemoryFaissLikeDB
import numpy as np


def test_inmemory_vectordb_add_and_search():
    db = InMemoryFaissLikeDB(dim=3)
    vecs = np.array([[1, 0, 0], [0, 1, 0]])
    metas = [{"id": 1}, {"id": 2}]
    db.add(vecs, metas)
    res = db.search(np.array([1, 0, 0]), k=1)
    assert res[0]["id"] == 1
