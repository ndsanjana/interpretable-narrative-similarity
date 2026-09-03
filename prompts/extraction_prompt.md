## 1. The prompt

System message:

```
You extract structured narrative information from folktales.
Output only JSON matching the schema. No commentary.
```

User message (substitute `{story_text}`):

```
Read the story and describe its narrative structure in abstract, role-based terms.

RULES

R1. Never write a proper name. Not of a person, a place, a people, a god, or a
    work. Use role descriptions instead: "the youngest daughter", "the
    supernatural suitor", "a talking helper animal".
R2. Never describe the setting. No places, countries, landscapes, seasons, eras,
    or cultural and religious markers. Numbers that carry story structure
    ("three brothers", "the third night") are structure, not setting: keep them.
R3. Every string you write is later read on its own, with no other text beside
    it. Write each string so it makes sense alone. No pronouns pointing outside
    the string: no bare "he", "she", "it", "they", "the same", "then", "there".
    Name the acting role inside the string.
R4. Report what happens, in plain reusable wording. Only abstract_theme is
    allowed to be abstract commentary.
R5. If the text is not a story, do not invent one. See DEGENERATE INPUT.

FIELDS

narrative_status: exactly one of
  full_narrative  a complete story: a situation, a complication, an ending
  fragment        very short: an anecdote, riddle, joke, verse, or a piece of a story
  composite       several self-contained episodes strung together under one title
  non_narrative   not a story: a bibliography, a list of titles, an editorial note

abstract_theme: 3 to 8 phrases, 3 to 8 words each, most important first.
  Concepts only. No actors, no events, no setting. Cover different dimensions:
  moral, emotional, relational, existential.

actors: the 2 to 5 agents the plot needs (1 is allowed for a fragment, never
  more than 6). Each actor has
    id          1, 2, 3, ... in order of importance
    role        an abstract role description, 2 to 6 words, no names
    motivation  what this actor wants or fears, 3 to 10 words, a full phrase
  In a composite, collapse recurring one-off figures that serve the same
  function into one collective role, such as "a series of credulous strangers".

central_conflict:
    summary          one sentence, 10 to 30 words: who or what is set against
                     whom or what, and over what
    protagonist_ids  actor ids on the protagonist side
    antagonist_ids   actor ids on the opposing side
  Either list may be empty when the opposing force is not an actor.

course_of_action: the major events in story order. 5 to 15 entries for a full
  narrative, 2 to 5 for a fragment, up to 25 for a composite. Each entry has
    id           1, 2, 3, ... in story order, no gaps
    description  one clause, 5 to 20 words, naming the acting role. Factual.
    function     the narrative function in 2 to 4 words, in plain reusable
                 wording: "impossible task set", "magical helper intervenes",
                 "false claimant exposed", "pursuer outwitted"
    actor_ids    the actors who act or are acted upon in this event
    caused_by    ids of EARLIER actions that DIRECTLY cause this one. Empty if
                 none. Never list an action just because it came before. Most
                 actions have 0 or 1 direct causes.

outcome: final states only, 1 to 5 entries. No intermediate states. Each has
    description  5 to 20 words
    kind         "resolution" (how the central conflict ends),
                 "fate" (where an actor ends up), or
                 "moral" (the lesson the tale states or implies)
    actor_ids    actors this outcome applies to; empty for a general moral

plot_type: exactly one of
  Overcoming the Monster  a threatening force is confronted and destroyed
  Rags to Riches          a low or despised figure rises to worth and fortune
  Quest                   a journey undertaken to obtain or achieve one thing
  Voyage and Return       a passage into a strange world and a changed return
  Comedy                  obstruction and confusion ending in union and pardon
  Tragedy                 a flaw or transgression ending in ruin or death
  Rebirth                 a figure held in a bad state is released or redeemed
  Trickster               deception, cunning, or folly drives the plot
  Cautionary Tale         a fable or exemplum where conduct brings its due result
  Other                   none of the above fits

DEGENERATE INPUT

Ignore editorial apparatus wherever it appears: source notes, translator and
editor credits, bibliographies, lists of related titles, "return to" links,
and commentary about the tale. Extract only from the story itself.

If nothing but apparatus remains, set narrative_status to "non_narrative",
plot_type to "Other", central_conflict.summary to "", and return empty lists
for abstract_theme, actors, course_of_action, and outcome.

If the text is a fragment, extract fewer items rather than inventing them.
Never add an actor, an action, or an outcome the text does not support.

If the text is a composite, keep all episodes in one course_of_action and let
caused_by show where an episode starts fresh instead of continuing the one
before it.

EXAMPLE

Story: A poor woodcutter's third son is refused bread by his brothers. An old
woman he feeds on the road gives him a whistle. When a giant blocks the way,
the boy blows the whistle, the giant falls asleep, and the boy carries the
giant's gold home.

{
  "narrative_status": "full_narrative",
  "abstract_theme": ["kindness repaid by unexpected aid",
                     "the least regarded sibling prevailing",
                     "cunning defeating brute force"],
  "actors": [
    {"id": 1, "role": "the youngest and poorest son",
     "motivation": "to survive and provide for himself"},
    {"id": 2, "role": "the withholding elder brothers",
     "motivation": "to keep the little they have"},
    {"id": 3, "role": "a disguised supernatural helper",
     "motivation": "to reward whoever shows kindness"},
    {"id": 4, "role": "a giant blocking the road",
     "motivation": "to keep travellers and treasure from passing"}
  ],
  "central_conflict": {
    "summary": "The youngest son, denied support by his brothers, must get past a giant who blocks the only road.",
    "protagonist_ids": [1],
    "antagonist_ids": [2, 4]
  },
  "course_of_action": [
    {"id": 1, "description": "The elder brothers refuse the youngest son any bread.",
     "function": "aid denied by kin", "actor_ids": [2, 1], "caused_by": []},
    {"id": 2, "description": "The youngest son shares his food with the disguised helper on the road.",
     "function": "kindness to a stranger", "actor_ids": [1, 3], "caused_by": []},
    {"id": 3, "description": "The disguised helper gives the youngest son a whistle.",
     "function": "magical gift received", "actor_ids": [3, 1], "caused_by": [2]},
    {"id": 4, "description": "A giant blocks the youngest son's road.",
     "function": "obstacle confronts hero", "actor_ids": [4, 1], "caused_by": []},
    {"id": 5, "description": "The youngest son blows the whistle and the giant falls asleep.",
     "function": "magical object defeats obstacle", "actor_ids": [1, 4], "caused_by": [3, 4]},
    {"id": 6, "description": "The youngest son carries the sleeping giant's gold home.",
     "function": "treasure won", "actor_ids": [1], "caused_by": [5]}
  ],
  "outcome": [
    {"description": "The youngest son returns home wealthy.",
     "kind": "resolution", "actor_ids": [1]},
    {"description": "Kindness freely given is repaid when it is needed.",
     "kind": "moral", "actor_ids": []}
  ],
  "plot_type": "Rags to Riches"
}

Now do the same for this story.

Story:
{story_text}
```

---

## 2. Schema summary

Full models in `src/schemas.py`. Seven top-level keys:

| key | type | node relevance |
| --- | --- | --- |
| `narrative_status` | enum (4) | not a node; a routing and filtering flag |
| `abstract_theme` | list[str], 3 to 8 | Theme nodes (text = the phrase) |
| `actors` | list[Actor{id, role, motivation}] | Actor nodes (text = `role`) and Motivation nodes (text = `motivation`) |
| `central_conflict` | Conflict{summary, protagonist_ids, antagonist_ids} | one Conflict node (text = `summary`) plus signed edges to Actor nodes |
| `course_of_action` | list[Action{id, description, function, actor_ids, caused_by}] | Action nodes (text = `description`, second coarse channel = `function`), temporal edges from id order, causal edges from `caused_by`, actor edges from `actor_ids` |
| `outcome` | list[Outcome{description, kind, actor_ids}] | Outcome nodes (text = `description`), typed by `kind`, with actor edges |
| `plot_type` | enum (10) | a learned categorical embedding on the Story node |

Every cross-reference in the record is an integer id. `caused_by` ids point to
`course_of_action` ids strictly earlier in the list. All `*_ids` fields point to
`actors[].id`. There is no string join anywhere in the record.

Settings are absent by design and there is no field that could carry one.

---
