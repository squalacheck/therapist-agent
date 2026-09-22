-- The clickable chips on the new-chat screen.
--
-- Open WebUI stores these in `config`, not in an environment variable, and
-- reads env only on first boot — so this writes the row directly, the same
-- way `make set-voice` and `make backdrop` do.
--
-- "Let's end for now" is the important one: it is the exact phrase
-- session.is_end_request() matches, so clicking it closes the session out
-- and stops the stack. Change the wording here and you must change the
-- regex there too.
UPDATE config
   SET value = '[
        {"title": ["Pick up where we left off", "what has changed since last time"],
         "content": "Pick up where we left off. What was I working on, and what should I be paying attention to today?"},
        {"title": ["Today''s list", "what should I actually do today"],
         "content": "What is on my practice list today, and where should I start?"},
        {"title": ["Something happened", "I want to think it through"],
         "content": "Something happened and I want to think it through properly before I decide what it means."},
        {"title": ["Help me say this", "I need to raise something difficult"],
         "content": "I need to raise something difficult with someone close to me and I do not want it to turn into the usual fight. Help me work out how to say it."},
        {"title": ["What am I missing", "push back on my account"],
         "content": "Here is my version of what happened. Tell me what I might be missing, and what it could have looked like from their side."},
        {"title": ["Let''s end for now", "save and close out"],
         "content": "let''s end for now"}
       ]'::json,
       updated_at = extract(epoch from now())
 WHERE key = 'ui.prompt_suggestions';
