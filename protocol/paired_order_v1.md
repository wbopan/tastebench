# Paired-order full-prefix model evaluation protocol, version 1

This protocol measures binary technical-decision judgment under both deterministic
option orderings. Each model sees the published option order and its exact reverse. The
released `choices` column already carries the evaluated order, so no shuffle seed enters
the evaluation. The task, candidate wording, and pre-decision evidence are otherwise
identical, and the correct letter is recomputed after ordering.

The model receives the `prefix_text` column of the released question, which renders the
trajectory through the decision fork. Credential-shaped text is redacted upstream, before
publication, so the released prefix needs no further treatment. The complete published
prefix is used when the request fits the configured input limit. An overflow preserves the
task setup at the beginning and the maximal possible tail nearest the decision point, with
an explicit omission marker.

The preflight counter runs locally: it encodes the rendered messages with the tiktoken
`o200k_base` vocabulary and multiplies the count by a configurable safety multiplier,
because tokenizers differ behind compatible APIs. Both the conservative preflight count
and the provider-reported prompt-token count must remain within the input limit.

The prompt defines the decision and final answer format but does not direct a reasoning
method or request visible chain of thought. Provider-native reasoning controls belong to
the versioned measurement config. Private reasoning fields are never persisted.

Headline accuracy uses every published question as the denominator; request errors and
unparseable outputs count as incorrect. Paired reporting includes both single-order
accuracies, their presentation-level mean, the proportion correct in both orders, the
four paired outcomes, position-choice counts, and the same metrics within each release
cell.
