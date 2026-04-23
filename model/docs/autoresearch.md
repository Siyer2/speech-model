Your goal is autonomously improve this model. The definition of an improvement is:
1) Improves the val/loss AND val/cer AND val/cer_errors AND the val/cer_target_word
2) Does NOT increase the model size

Follow this loop:
1) Create a branch off autoresearch/improve-model
2) Analyse the current code and come up with a change that should clearly lead to an improvement. 
3) Train the model with your change using `make train NAME="informative-name" NOTE="some description"`
4) After 3 epochs, determine if there has been a step change improvement because of your changes OR, if it is on par with baseline, but the rate of change is significant you may allow it to run for a little longer. EARLY STOP if there is no clear improvement. EARLY STOP if it is clear this is an improvement. Do not waste time running an experiment for too long.
5) ONLY keep the code if it has led to a CLEAR improvement
6) If it has led to an improvement, update model/docs/autoresearch-results.md with the new point (with a name) on the line graph, add a concise bullet point mapping the name to a description and merge back to the autoresearch/improve-model branch.
7) Begin with step 1 of the loop again - the improvement becomes the new baseline.

NOTE: you can use the results in model/wandb/run-20260222_114103-4i0w36iw as the initial baseline to work off. It does not have val/cer_target_word, however ONLY for this run you may assume improving val/loss, val/cer and val/cer_errors improves val/cer_target_word. Subsequent comparison must stick to the original improvement definition.

Always ensure that you are maximising available resources - GPU and CPU utilisation must be as close to maximised as possible.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~20 minutes then you can run approx 3/hour, for a total of about 24 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!