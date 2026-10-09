# 4d.0: candidate sources for coaches_history (2016-2026), and what their terms allow

**Status: STOPPED for your review. Nothing has been collected.** To see one file's format I downloaded a 4 KB header sample (Source 3) and
deleted it straight away. The quotes below were pulled from each page on 2026-10-09 by a web-fetch tool, so each is listed with its URL.
Check the exact wording at the URL before deciding.

The table needs one row per team, season and role (head_coach, offensive_coordinator, play_caller), with UNKNOWN where nothing reliable exists.

## Summary

| # | Source | Covers | Granularity | Automated collection | Use in a prediction model | Redistribution | Stable enough to rebuild from | Recommendation |
|---|---|---|---|---|---|---|---|---|
| 1 | nflverse schedules (already in our raw DB) | head coach | per game, 2016-2024 100% filled | yes (already pulled) | no clause found | repo CC-BY-4.0 (nflverse-data); schedules originate in nfldata, which states no license | yes (our pull is stored and fingerprinted) | **use for head_coach** |
| 2 | Wikipedia team-season pages | head coach, offensive coordinator (title only) | per season | allowed through the API, within the bot / User-Agent / etiquette policies | no clause either way | CC BY-SA 4.0 (attribution, share-alike on anything redistributed) | pages change; pin the revision IDs | **use for offensive_coordinator** |
| 3 | Sam Hoppen, `NFL_public/data/all_playcallers.csv` (GitHub) | offensive play caller, defensive play caller, head coach | **per game**, 1999-2026 | file download (no site terms involved) | **no license at all**, so no permission is granted | none granted | actively corrected (9 commits Jan-Oct 2026, some changing past seasons): pin the commit SHA | **best play_caller source, but needs the author's permission first** |
| 4 | Pro Football Reference / Sports Reference | head coach, coordinators | per season | **prohibited** | **prohibited** | single pages yes, en masse no | n/a | **do not use** |
| 5 | AP "Who calls plays for every NFL team in <year>" (on ABC / Disney station sites) | play caller | per season, recent seasons only | **prohibited** (Disney terms) | **prohibited** (Disney terms) | prohibited | n/a | spot checks by hand only, never a source |
| 6 | Wikidata | head coach (sparse) | spans | API | no clause | CC0 (public domain) | yes (revisions) | fallback only; adds nothing over 1 |

## Details and quoted terms

### 1. nflverse schedules: `home_coach` / `away_coach`
- **What it has.** Defined in nfldata DATASETS.md: `home_coach` "Name of the head coach of the home team", `away_coach` "Name of the head coach of
  the away team" (https://raw.githubusercontent.com/nflverse/nfldata/master/DATASETS.md). In our raw DB both are filled for every
  2016-2024 regular-season game (2,367 of 2,367). Being per game, it captures interim head coaches.
- **Terms.**
  - nflreadr (https://nflreadr.nflverse.com/): "NFL data accessed by this package belong to their respective owners, and are governed by
    their terms of use."
  - The nflverse-data repository is labelled "CC-BY-4.0 license" (https://github.com/nflverse/nflverse-data).
  - The nfldata repository, where the schedules originate, shows no license. Its DATASETS.md credits Pro Football Reference for its draft
    picks, trades and rosters, but names no source for the coach columns.
- **Open point.** The head coach columns' provenance is not stated. Given the PFR terms below, you may want to ask nflverse where the
  coach names come from before relying on them for more than what nflverse already distributes.

### 2. Wikipedia team-season pages (e.g. https://en.wikipedia.org/wiki/2023_Kansas_City_Chiefs_season)
- **What it has.** The infobox and staff section list "Head coach: Andy Reid" and "Offensive coordinator: Matt Nagy", but no play caller
  (Reid called Kansas City's plays). So it fills offensive_coordinator, not play_caller.
- **Terms** (https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use, in effect June 7, 2023):
  - **License.** Text is under the "Creative Commons Attribution-ShareAlike 4.0 International License" ("CC BY-SA 4.0"). Reuse must be
    "properly attributed and the same freedom to reuse and redistribute is granted" to derivative works.
  - **Automated access.** Prohibited are "Engaging in automated uses of the Project Websites that are abusive or disruptive of the services"
    and "Disrupting the services by placing an undue burden on an API, Project Website or the networks or servers". Section 12: "By using our
    APIs, you agree to abide by all applicable policies governing the use of the APIs" (User-Agent policy, Robot policy, API:Etiquette).
  - **AI / ML.** No clause found.
- **In practice.** About 352 pages (32 teams x 11 seasons), read through the API with a descriptive User-Agent at a polite rate. Store the
  revision ID of each page read. Attribute the source, and keep any redistributed coaches_history CSV under CC BY-SA.

### 3. Sam Hoppen, `all_playcallers.csv` (https://github.com/samhoppen/NFL_public/blob/main/data/all_playcallers.csv)
- **What it has.** Columns `season, week, team, game_id, off_play_caller, def_play_caller, head_coach`, one row per team-game since 1999
  (game_id in nflverse format). This is the only source found with the actual play caller, per game, for every season. The Patton
  Analytics play-caller study credits it: "thanks to Sam Hoppen for compiling a list of offensive and defensive play callers for every
  NFL team dating back to 2013" (https://pattonanalytics365.substack.com/p/evaluating-play-callers-in-the-nfl).
- **Terms.** The repository shows no license, and its README says only "A housing place for some public NFL data and code!". With no
  license, nothing grants reuse rights, so the safe reading is that we need the author's permission (by email, or an issue on the repo)
  before using it in the model.
- **Stability.** It is maintained and still corrected. Recent commits: "so long davis webb", "Demeco calling plays" (Oct 8, 2026), "update
  CAR 2022 Weeks 1-5" (May 20, 2026), "fix NYG OPC in 2025" (Mar 6, 2026). Past seasons do change, so with permission we would pin one
  commit SHA, store the file's hash, and update only deliberately: the lesson from Sumer.

### 4. Pro Football Reference / Sports Reference: do not use
- **Terms of Use** (https://www.sports-reference.com/termsofuse.html; Last Updated May 19, 2023). Prohibited are:
  - "without our express written permission, use any automated means to access or use the Site ... including scripts, bots, scrapers, data
    miners, or similar software"
  - "to create any database, archive, or other data store that competes with or constitutes a material substitute"
  - use "for purposes of training, fine-tuning, prompting, or instructing artificial intelligence models ... (ii) supporting machine
    learning methods used to predict, classify, label, or score inputs into the models"
- **Data Use page** (https://www.sports-reference.com/data_use.html): "For some of our datasets, our licenses completely preclude any
  redistribution of the data", and "Please do not attempt to aggressively spider data from our web sites".
- **Why not.** Feeding a prediction model is exactly the prohibited ML use.

### 5. AP "Who calls plays for every NFL team in <year>", syndicated on ABC station sites: not a source
- **Example.** https://abc7news.com/post/calls-plays-every-nfl-team-2025-what-know/17777918/. The 2026 version is already the cited source in
  data/coaches_seed.csv (Phase 1.7, entered by hand). Searches found the 2024, 2025 and 2026 editions, not 2016-2023.
- **Disney Terms of Use** (https://disneytermsofuse.com/english/) prohibit:
  - to "access, monitor, copy or extract the Disney Products using a robot, spider, script, or other automated means, including ... for
    the purposes of creating or developing any AI Tool, data mining or web scraping"
  - use "in connection with any use, creation, development, modification, prompting, fine-tuning, training, testing, benchmarking or
    validation of any artificial intelligence or machine learning tool, model, system"
- **Use.** Facts read by a person for spot checks only, never collected.

### 6. Wikidata
- **License.** Structured data "is released into the public domain under" Creative Commons Zero (https://www.wikidata.org/wiki/Wikidata:Licensing).
- **Coverage.** Head coaches only, and patchy; it adds nothing over source 1.

## Proposed plan, for your decision
- **head_coach** from source 1 (already in hand, per game).
- **offensive_coordinator** from source 2 (Wikipedia through the API, revision IDs stored, CC BY-SA kept on anything redistributed).
- **play_caller** from source 3, only if Sam Hoppen grants permission (commit SHA pinned). Without it, play_caller is UNKNOWN wherever it is
  not the head coach or the coordinator by a sourced statement, and the 4d.1 prior falls back to head coach / coordinator.
- Every row carries its source and revision; rows nobody can fill are UNKNOWN, never guessed; UNKNOWN counts are printed per season (plan 4d.0).

Nothing will be collected until you approve sources, and for source 3 until the author's permission is in hand.
