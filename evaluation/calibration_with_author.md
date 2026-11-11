# Calibrating Round 2 and Round 3 against human labels (2026-11-11)

Labels: real_backend: A, B, self (17 labelled, 17 judged); real_data: A, B, self (12 labelled, 12 judged); real_frontend: A, B (100 labelled, 10 judged); real_ml: A, B (100 labelled, 10 judged); real_sre: A, B, self (11 labelled, 11 judged). A candidate's grade is the mean of these labellers' grades; Round 3 is scored on the labelled candidates Claude judged (on a sheet covering every resume, Round 1's top 10).

Agreement between labellers:
- A vs B: weighted kappa 0.93 (almost perfect), 240 candidates
- A vs self: weighted kappa -0.15 (no better than chance), 40 candidates
- B vs self: weighted kappa -0.12 (no better than chance), 40 candidates

Rounded mean grades: 14 x 2 (would interview), 33 x 1 (borderline), 13 x 0, over 60 candidates. An always borderline rule would agree on 0.55 of them with kappa 0, so agreement alone rewards saying borderline. Rules are therefore chosen on quadratic-weighted kappa, which corrects for chance.

## Round 1: the hybrid search's ranking of all 100 resumes

| Posting | labelled | P@10 | nDCG@10 | R@10 |
|---|---|---|---|---|
| real_backend | 17 | 0.40 | 0.64 | n/a |
| real_data | 12 | 0.50 | 0.95 | n/a |
| real_frontend | 100 | 1.00 | 0.98 | 0.45 |
| real_ml | 100 | 0.80 | 0.51 | 0.19 |
| real_sre | 11 | 0.90 | 0.94 | n/a |

R@10 is shown only where the labels cover every resume (n/a on shortlist sheets).

## Round 3: the hire / borderline / no hire rule

Kappa: quadratic-weighted, the call (2 / 1 / 0) against the rounded mean grade. Agreement: share where they are equal. Hire precision: share of Hire calls graded 1 or more. Missed: graded 1.5 or more, called no hire.

| Rules | mean kappa | agreement | hire precision | interviews missed | real_backend | real_data | real_frontend | real_ml | real_sre |
|---|---|---|---|---|---|---|---|---|---|
| hire at fit >= 70, borderline at fit >= 40 | 0.34 | 0.62 | 1.00 | 3 | 0.25 | 0.76 | 0.40 | 0.00 | 0.30 |
| hire at fit >= 70, borderline at fit >= 50 | 0.25 | 0.52 | 1.00 | 4 | 0.40 | 0.09 | 0.44 | 0.00 | 0.30 |
| hire at fit >= 80, borderline at fit >= 40 | 0.20 | 0.58 | n/a | 3 | 0.23 | 0.76 | 0.00 | 0.00 | 0.00 |
| hire at fit >= 90, borderline at fit >= 40 | 0.20 | 0.58 | n/a | 3 | 0.23 | 0.76 | 0.00 | 0.00 | 0.00 |
| hire at fit >= 80, borderline at fit >= 50 | 0.15 | 0.47 | n/a | 4 | 0.41 | 0.09 | 0.22 | 0.00 | 0.00 |
| hire at fit >= 90, borderline at fit >= 50 | 0.15 | 0.47 | n/a | 4 | 0.41 | 0.09 | 0.22 | 0.00 | 0.00 |
| hire: none missing, <= 0 partial; borderline: <= 0 missing **(current)** | 0.00 | 0.20 | n/a | 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

| Held out | Chosen on the other postings | Its score there | Current rules there |
|---|---|---|---|
| real_backend | hire at fit >= 70, borderline at fit >= 40 | 0.25 | 0.00 |
| real_data | hire at fit >= 70, borderline at fit >= 50 | 0.09 | 0.00 |
| real_frontend | hire at fit >= 70, borderline at fit >= 40 | 0.40 | 0.00 |
| real_ml | hire at fit >= 70, borderline at fit >= 40 | 0.00 | 0.00 |
| real_sre | hire at fit >= 70, borderline at fit >= 40 | 0.30 | 0.00 |

Held out, the rules chosen on the other postings did better than the current ones on 4 of 5 postings, worse on 0, the same on 1.

## Round 2: the fit score's ranking of Round 1's top 10

| Rules | mean ndcg10 | P@5 | real_backend | real_data | real_frontend | real_ml | real_sre |
|---|---|---|---|---|---|---|---|
| must-haves x2, partial = 0.5, unverified met = partial **(current)** | 0.83 | 0.88 | 0.66 | 0.89 | 0.99 | 0.65 | 0.96 |
| must-haves x2, partial = 0.5, unverified met = met | 0.83 | 0.88 | 0.66 | 0.89 | 0.99 | 0.65 | 0.96 |
| must-haves x3, partial = 0.5, unverified met = partial | 0.83 | 0.88 | 0.66 | 0.91 | 0.99 | 0.63 | 0.95 |
| must-haves x3, partial = 0.5, unverified met = met | 0.83 | 0.88 | 0.66 | 0.91 | 0.99 | 0.63 | 0.95 |
| must-haves x1, partial = 0.5, unverified met = partial | 0.83 | 0.84 | 0.66 | 0.86 | 1.00 | 0.65 | 0.96 |
| must-haves x1, partial = 0.5, unverified met = met | 0.83 | 0.84 | 0.66 | 0.86 | 1.00 | 0.65 | 0.96 |

| Held out | Chosen on the other postings | Its score there | Current rules there |
|---|---|---|---|
| real_backend | must-haves x2, partial = 0.75, unverified met = partial | 0.61 | 0.66 |
| real_data | must-haves x1, partial = 0.5, unverified met = partial | 0.86 | 0.89 |
| real_frontend | must-haves x2, partial = 0.5, unverified met = partial | 0.99 | 0.99 |
| real_ml | must-haves x3, partial = 0.5, unverified met = partial | 0.63 | 0.65 |
| real_sre | must-haves x2, partial = 0.5, unverified met = partial | 0.96 | 0.96 |

Held out, the rules chosen on the other postings did better than the current ones on 0 of 5 postings, worse on 3, the same on 2.

How to read this: a rule earns a change only if it beats the current rules on postings it was not chosen on. A handful of postings and labellers show a direction, not a significant result.
