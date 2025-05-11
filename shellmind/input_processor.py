class InputProcessor:
    def __init__(self):
        pass

    def process_query(self, raw_query: str) -> str:
        """Cleans and prepares the raw user query for the AI."""
        processed_query = raw_query.strip()
        return processed_query