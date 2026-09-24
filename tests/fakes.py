class FakeChatModel:
    """Stands in for a LangChain chat model in tests.

    Only supports the one thing the agent uses: with_structured_output(schema).invoke(messages).
    Each invoke() returns (or raises, if it is an Exception) the next scripted response.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.schemas = []
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
