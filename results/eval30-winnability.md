# Winnability census of eval30's never-solved tasks

Method: each eval30 task solved in at most 1 of 7–16 local runs (21 tasks) was compared with its upstream fix and hidden tests, used only as an answer key to judge, never shown to the model. *Unwinnable* means the tests require something the issue does not state. Task text, tests and patches are not reproduced here; the issues and fixes are public in the upstream repositories.

| Verdict | n | Tasks and general reason |
|---|---|---|
| Unwinnable: missing information | 5 | fastapi_13713 (parameter name differs from the issue's), fastapi_14583 (exact warning text), fastapi_14258 (exact error message), fastapi_14306 (unstated name and exact output format), rich_3942 (exact colours of new styles) |
| Unwinnable: out of scale | 7 | fastapi_14099 (large redesign plus new example files), fastapi_14609 (20-file migration), fastapi_14978 (title-only issue plus new example files), fastapi_15030 (large feature with new tutorials), fastapi_15661 (release script with exact output), fastapi_15800 (large new feature), fastapi_14371 (23 test files from a vague issue) |
| Underspecified | 6 | fastapi_14246 (issue is only links), fastapi_11355 (issue only links another project's change), rich_3180 (60-line rewrite with exact output defined by stated behaviour), rich_3772 (issue is two links), rich_4070 (tests depend on one specific implementation), rich_4079 (title-only issue) |
| Winnable | 3 | fastapi_14616 (detailed issue, 8-line fix), fastapi_13920 (3-line fix), rich_3063 (2-line fix) |

Failure modes on the winnable three: rich_3063 spends its calls on shell-quoting errors and never edits the target file; fastapi_13920 adds the right helper but never changes the code path that should call it; fastapi_14616 edits the wrong file or only part of the affected code.
