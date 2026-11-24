"""Connect video ranking to the catalog and optional review data."""

from ..database.repository import VideoRepository, prepare_database
from ..retrieval.search_engine import SearchEngine


class QueryService:
    def __init__(self, db_path, source):
        if (source / "features.npz").is_file():
            prepare_database(db_path, source)
        self.repository = VideoRepository(db_path)
        self.engine = SearchEngine(*self.repository.load_catalog())

    def search(self, key, metric, with_d, other_people, *, read_only=False):
        results, total = self.engine.rank(key, metric, with_d, other_people)
        result = {"query": self.engine.items[self.engine.index[key]],
                  "candidates": total, "results": results}
        if not read_only:
            ratings, count = self.repository.ratings_for_query(key)
            result["results"] = [
                dict(match, verdict=ratings.get(match["video"]["id"])) for match in results
            ]
            result["rating_count"] = count
        return result
