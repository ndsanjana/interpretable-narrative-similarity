## 1. The prompt

System message:

```
You are a careful analyst of narrative causation. You are given a story and a
fixed numbered list of its actions. You output only JSON matching the schema.
No commentary.
```

User message (substitute `{story_text}` and `{action_list}`):

```
Your only job is CAUSAL ANALYSIS over a fixed list of actions. The actions are
already extracted and numbered. You do not re-segment them, re-describe them,
or add to them. For every action id in the list you decide which EARLIER
actions actually cause it.

WHAT COUNTS AS A CAUSE

A parent action causes a child action if it does at least one of these:
  enable it            the child is only possible because the parent happened
  motivate it          the parent gives the actor the reason or the knowledge
  bring it about       the parent physically produces the child

An action that merely happened before the child is NOT a parent. Temporal
order is already recorded by the id numbers, so repeating it carries no
information.

THE ONE ERROR TO AVOID

Do not put the immediately previous action in the parent list just because it
is the previous action. That is the default answer and it is usually
incomplete and sometimes wrong. List the previous action only when it really
does enable, motivate or produce the child. Many actions have a genuine cause
that sits five or ten steps earlier, and many have no cause at all.

HOW TO SEARCH

Step 1. Before assigning any parents, read the whole action list and mark
every action that establishes a STANDING CONDITION: a magic object handed
over, a power or wish granted, a promise or an oath made, a rule or a
prohibition stated, a secret learned, a bargain struck. A standing condition
stays in force after the action that creates it. Report these in
standing_conditions.

Step 2. For every action, scan ALL earlier actions, not just the last one, and
run these four checks:

  (a) STANDING ENABLING CONDITION. Is this action only possible because of a
      standing condition from step 1? The wish is granted by the promise made
      long before. The hero can carry the treasure because of the bag that
      never fills. The sentence is passed because of the prohibition stated at
      the start. List the action that created the condition as a parent, every
      time the condition operates, not only the first time.

  (b) SETUP AND PAYOFF. Does this action discharge something set up much
      earlier? A promise kept, a debt collected, a threat carried out, a
      secretly acquired fact finally used, a name explained. List the setup
      action as a parent even if it is far back.

  (c) CONVERGENCE. Is this a trial, a judgment, a revelation, a recognition,
      a proof, a reward, a punishment, or the final resolution? These are
      where independent strands meet. They usually have three to five parents:
      every piece of evidence, every promise, and every deed that the outcome
      rests on. A judgment scene with one parent is almost always wrong.

  (d) PARALLEL STRANDS. Does this action run alongside another action rather
      than after it? Several friends each doing one part of one scheme, three
      stages of one journey, two scenes happening at the same time in two
      places. Actions like these usually SHARE a parent, the action that set
      the scheme or the journey going. They do not cause each other.

Step 3. Some actions have NO parent. Say so with an empty list. Scene setting,
background states, the opening of an independent episode, an arrival the story
presents as coincidence ("some time after this a man was plowing"), and a
character acting for reasons the story does not connect to anything earlier
are all genuinely uncaused inside this action list.

RULES

R1. Every action id in the list appears exactly once in links, in ascending
    order. No id is skipped and none is invented.
R2. Every parent id is strictly smaller than the action id it is a parent of.
    An action never causes itself and never causes an earlier action.
R3. Parent lists may be empty, may hold one id, or may hold several. There is
    no upper limit, but every id you list must be a cause you can defend.
R4. Precision matters as much as coverage. If you are not willing to say the
    child would not have happened, or would have happened differently, without
    the parent, leave the parent out.
R5. Most tales end up with between 1.0 and 1.4 parent links per action, and a
    fifth to a third of actions carry more than one parent. If your answer is
    a straight chain 1, 2, 3, 4, you have not searched. Straight chains do
    occur, in repetitive trick tales and cumulative formula tales, but they
    are the minority.
R6. why is one short clause per action naming the reason for each parent, or
    the reason there is none. Twenty-five words at most. Never a restatement
    of the action.

WORKED EXAMPLE 1: convergence, standing conditions and parallel strands

Tale gist: an old man gives a pious shepherd son a bag that never fills and a
pelt that makes him invisible. A king cannot explain why eleven of his twelve
daughters wear out their shoes every night, and offers the youngest daughter
to whoever solves it. The shepherd son follows the daughters invisibly through
three forests to a spirit's ball, collecting a token in each forest, and wakes
the youngest daughter so she can witness it. The tokens and her testimony
convict the eleven.

Action list:
  1. The gray old man grants a bag that never fills and an invisible pelt to
     the pious shepherd son
  2. The pious shepherd son leaves his pasture and travels to the capital city
  3. The king has twelve daughters, eleven of whom wear many shoes nightly,
     causing cost and rumors
  4. The king offers his youngest daughter to whoever solves the mystery of
     the shoe-wearing
  5. Many suitors arrive, are ridiculed by the daughters, and withdraw in shame
  6. The pious shepherd son, wearing the invisible pelt, slips into the
     daughters' bedroom at night
  7. The mysterious spirit enters, wakes the daughters, and they prepare by
     putting shoes into a bag
  8. The pious shepherd son awakens the youngest daughter, who then agrees to
     join them
  9. The mysterious spirit places a basin; the daughters rub its contents,
     wings grow, and they all fly
  10. They fly to a copper forest, drink, and the shepherd son collects a cup
      and leaves a copper twig in his bag
  11. They fly to a silver forest, drink, and the shepherd son collects a
      silver cup and leaves a silver twig in his bag
  12. They fly to a golden forest, drink, and the shepherd son collects a
      golden cup and leaves a golden twig in his bag
  13. They reach a mountain cliff; the mysterious spirit opens a passage, and
      they enter a hall of fairy-like splendor
  14. In the hall, fairy-youths appear, music plays, and the group enjoys a
      ball before returning home
  15. The king learns of the events, summons the daughters, and evidence
      proves their guilt
  16. The king fulfills his promise; the youngest daughter marries the
      shepherd son, and the eleven daughters are burned

Correct answer:
  standing_conditions: 1 (bag that never fills, invisible pelt),
                       4 (the king's standing offer)
  1  parents []            opening gift, nothing earlier
  2  parents [1]           he travels because he has been equipped
  3  parents []            background state of the court, uncaused here
  4  parents [3]           the offer answers the unexplained nightly wear
  5  parents [4]           suitors come because of the offer
  6  parents [1, 4]        the pelt makes entry possible, the offer is the motive
  7  parents []            the spirit's nightly visit happens with or without him
  8  parents [6, 7]        he is in the room and the others are being roused
  9  parents [7]           the same visit continues; action 8 is his own aside
  10 parents [1, 9]        the flight carries them there, the bag holds the twig
  11 parents [1, 9]        second forest of the same flight, not caused by the first
  12 parents [1, 9]        third forest of the same flight, not caused by the second
  13 parents [9]           the same flight reaches the cliff
  14 parents [13]          the ball follows entry into the hall
  15 parents [6, 8, 10, 11, 12]  five strands converge: he was there, the
                           woken sister testifies, three tokens are produced
  16 parents [4, 15]       the standing offer is discharged once guilt is proved

Note what is happening at 15 and at 10, 11, 12. Node 15 is a judgment: five
parents, not one. Nodes 10, 11 and 12 are three stages of one flight, so they
share the parents 1 and 9 instead of forming a chain. Only four of the sixteen
actions have the previous action as their sole parent.

WORKED EXAMPLE 2: a standing grant and a long-range payoff

Tale gist: a woodman spares a fairy's oak and is promised three wishes. He
forgets, then idly wishes for a hog's pudding, which appears. His wife, angry
at the wasted wish, wishes it onto his nose. He spends the third wish removing
it, and the promise is used up.

Action list:
  1. The woodman chops a huge oak when a fairy appears.
  2. The woodman consents to spare the tree, and the fairy promises three
     wishes.
  3. The woodman forgets the fairy by evening.
  4. The woodman wishes for a link of hog's pudding at night.
  5. A rustling in the chimney brings the pudding to the woodman's feet.
  6. The woodman tells his wife about the pudding.
  7. The wife, incensed, wishes the pudding to attach to her nose.
  8. The pudding sticks tightly to the wife's nose, impossible to remove.
  9. The woodman wishes the pudding off, fulfilling his third wish.
  10. The fairy's promise of wishes ends, leaving the woodman without further
      wishes.

Correct answer:
  standing_conditions: 2 (the promise of three wishes)
  1  parents []          opening scene
  2  parents [1]         the promise is the price of sparing the tree
  3  parents []          forgetting is not caused by the promise itself
  4  parents [3]         he wishes idly because he has forgotten what it costs
  5  parents [2, 4]      the wish takes effect only because of the promise
  6  parents [5]         he reports what appeared
  7  parents [4, 6]      she is angry at the wasted first wish, which she now
                         hears about
  8  parents [2, 7]      the second wish takes effect through the promise
  9  parents [2, 8]      the third wish takes effect through the promise
  10 parents [2, 4, 7, 9]  the grant of three is exhausted by the three wishes

The promise at action 2 is a parent five separate times, because it is in force
the whole time and nothing else makes the wishes work. Action 10 is a
convergence: it is the sum of the grant and all three wishes that spent it.

NOW ANALYSE THIS TALE

TALE
{story_text}

ACTION LIST
{action_list}

Return JSON with standing_conditions and links. links has one entry for every
action id above, in ascending order.
```

## 2. Output schema

The runner builds this per tale, with `n` substituted by that tale's action
count, and sends it as an OpenAI `json_schema` response format (falling back to
`guided_json` and then to a bare `json_object`, exactly as `extract.py`
does).

```
{
  "type": "object",
  "properties": {
    "standing_conditions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "action_id": {"type": "integer"},
          "condition": {"type": "string"}
        },
        "required": ["action_id", "condition"],
        "additionalProperties": false
      }
    },
    "links": {
      "type": "array",
      "minItems": n,
      "maxItems": n,
      "items": {
        "type": "object",
        "properties": {
          "action_id": {"type": "integer"},
          "parents": {"type": "array", "items": {"type": "integer"}},
          "why": {"type": "string"}
        },
        "required": ["action_id", "parents", "why"],
        "additionalProperties": false
      }
    }
  },
  "required": ["standing_conditions", "links"],
  "additionalProperties": false
}
```

`minItems` and `maxItems` pin the links array to one entry per action. Not
every grammar backend honours array length bounds either, so the runner's
probe walks json_schema pinned, json_schema unpinned, guided_json pinned,
guided_json unpinned, and finally a bare json_object, and records which one it
settled on in the stats sidecar.

Integer ranges are deliberately not expressed in the schema. Grammar backends
differ in whether they honour `minimum` and `maximum` on integers, and a
silently ignored bound is worse than no bound. The runner validates ids in
Python instead and counts every repair, so the rates are measured:

- parent id not in `1..n`            dropped, counted as `unknown_parent`
- parent id equal to the action id   dropped, counted as `self_loop`
- parent id greater than the action id  kept out of `caused_by` and recorded
  separately in `forward_parents`, counted as `forward_parent`
- duplicate parent id                deduplicated, counted as `duplicate_parent`
- missing or extra action entry      filled with an empty parent set or
  dropped, counted as `missing_action` / `extra_action`
- any cycle that survives the above  broken by dropping the most recently
  added edge, counted as `cycle_break`
