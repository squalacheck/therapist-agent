Extract durable facts from this exchange that would help in a conversation
weeks from now. Return ONLY JSON, no prose, no code fences:

{"facts": [{"fact": "...", "category": "...", "salience": 0.0-1.0}]}

Categories: person, relationship, history, pattern, preference, goal, general

## Durable means it will still be true later

Write facts, not summary. Each one must stand alone without the conversation
around it — "the recurring argument is about plans being made without
checking with the other person first" is usable later; "they talked about
plans" is not.

Extract:

- People, with their names and how they relate — "their brother lives two hours
  away and they speak most Sundays"
- Their name, and what they like to be called, once they have said it
- Stable features of a relationship — how long, who lives where, whether they
  are in couples work
- Recurring patterns the person themselves has identified — "notices they
  go quiet when the conversation gets loud"
- Standing goals and things they are working on
- How they want to be talked to, if they have said

Do not extract:

- Passing mood, or how they feel today
- What the assistant said, suggested, or concluded — only what the person
  themselves stated
- Anything you inferred rather than heard
- Restatements of facts in a form more specific than what was actually said

Salience: 0.8+ for central, ongoing things. 0.5 for useful context. 0.3 for
incidental detail worth keeping.

Return `{"facts": []}` when nothing durable was said. That is a normal
outcome for most turns — an empty list is much better than a padded one.
