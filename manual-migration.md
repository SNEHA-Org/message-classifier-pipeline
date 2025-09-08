Step 1:
Clone the output table - glific_messages_funnel

Step 2:
Reset and refreah all streams through Dalgo

Step 3: 
Run dbt full refresh for this model only to ensure all messages are pulled in

Step 4:
UPDATE prod.glific_messages_funnel AS g
SET
  body_final           = c.body_final,
  body_final_phonetic  = c.body_final_phonetic,
  question_type        = c.question_type,
  theme                = c.theme
FROM prod."glific_messages_funnel_clone" AS c
WHERE c.id = g.id
  AND (
    c.body_final          IS DISTINCT FROM g.body_final OR
    c.body_final_phonetic IS DISTINCT FROM g.body_final_phonetic OR
    c.question_type       IS DISTINCT FROM g.question_type OR
    c.theme               IS DISTINCT FROM g.theme
  );

  Step 5: (Optional) Run Github Actions to catch up on classifying new messages (This is scheduled at 9pm IST every day and can be skipped)





