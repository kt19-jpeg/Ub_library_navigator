from pydantic import BaseModel, Field

FAITHFULNESS_SYSTEM = """You check whether a one-line explanation about a library search result is
grounded in that result's real metadata, or whether it invents something not actually present.

Given the candidate's real title, author, subject, and description, and the stated reason for
recommending it, return grounded=true only if every claim in the reason is actually supported by
those fields. A reason that stays general or reasonably infers from the topic is fine; a reason
that states something as fact which isn't in the given fields (a specific finding, a date, a claim
about content not in the description) is not grounded."""


class FaithfulnessVerdict(BaseModel):
    grounded: bool = Field(
        description="True only if every claim in the reason is supported by the candidate's real fields."
    )


def judge_faithfulness(model, candidate_view: dict, reason: str) -> bool:
    """LLM-judge: is `reason` grounded in `candidate_view`'s real fields? Never raises: a model
    failure counts as ungrounded (fails closed) rather than crashing the whole eval run."""
    try:
        structured = model.with_structured_output(FaithfulnessVerdict)
        verdict = structured.invoke(
            [
                ("system", FAITHFULNESS_SYSTEM),
                ("human", f"candidate: {candidate_view}\nreason: {reason}"),
            ]
        )
        if isinstance(verdict, dict):
            verdict = FaithfulnessVerdict.model_validate(verdict)
        if not isinstance(verdict, FaithfulnessVerdict):
            return False
        return verdict.grounded
    except Exception:
        return False
