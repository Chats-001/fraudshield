# Cost model

FraudShield converts probability scores into review decisions with an explicit validation-only
objective:

`FP count × review cost + missed fraud dollars × loss rate`

The default review cost is `$5`, loss rate is `1.0`, and minimum fraud recall is `0.80`. These are
transparent demonstration assumptions—not estimates of any bank's costs. CLI flags and environment
variables make all three adjustable. The chosen threshold is the least-cost candidate satisfying
the recall constraint; ties prefer the higher threshold to avoid unnecessary reviews.

This simple model excludes recovery delays, chargeback fees, investigation capacity, customer
friction, and fraud amounts not represented by transaction `Amount`. A real implementation should
estimate those quantities from operations data and evaluate uncertainty/scenarios.

