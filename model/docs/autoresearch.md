Your goal is autonomously improve this model. The definition of an improvement is:
1) Improves the val/loss AND val/cer AND val/cer_errors AND the val/loss-target-word
2) Does NOT increase the model size

Follow this loop:
1) Create a branch off autoresearch/improve-model
2) Analyse the current code and come up with a change that should clearly lead to an improvement
3) Train the model with your change using `make train NAME="informative-name" NOTE="some description"`
4) After 3 epochs, determine if there has been a step change improvement because of your changes OR, if it is on par with baseline, but the rate of change is significant you may allow it to run for a little longer. EARLY STOP if there is no clear improvement. EARLY STOP if it is clear this is an improvement. Do not waste time running an experiment for too long.
5) ONLY keep the code if it has led to a CLEAR improvement
6) If it has led to an improvement, update model/docs/autoresearch-results.md with the new point (with a name) on the line graph, add a concise bullet point mapping the name to a description and merge back to the autoresearch/improve-model branch.
7) Begin with step 1 of the loop again - the improvement becomes the new baseline.

NOTE: you can use the results in model/wandb/run-20260222_114103-4i0w36iw as the initial baseline to work off. It does not have val/loss-target-word, however ONLY for this run you may assume improving val/loss, val/cer and val/cer_errors improves val/loss-target-word. Subsequent comparison must stick to the original improvement definition.

Do not ask me for help — exhaust your options first.