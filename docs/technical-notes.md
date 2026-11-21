# TheGavel technical notes

The full engineering write-up behind [the short article](https://michaelinghilterra.com/writing/your-judge-does-not-need-to-talk): what was measured, how, and what surprised me. The job postings used in testing stay private; everything here is aggregate numbers.

## Why TheGavel exists

I use a local model to screen job postings. It reads a posting plus my profile and rates it on seven things: fit, level, location, pay, and so on. Then code turns the ratings into one score from 0 to 5. The model was writing about 575 tokens per posting to produce what is, in the end, seven digits.

Every inference engine I could run is tuned for the opposite job: a short prompt and a long answer, measured in tokens per second. A judge is a long prompt (here about 25,000 tokens of rubric, profile and posting) and a tiny answer. So I built a small engine for the judge's shape of work, [TheGavel](https://github.com/michaelinghilterra-creator/TheGavel) (the command is `gavel`), and measured it against what I already had.

## The idea: read the probabilities, don't sample a digit

When the model reaches the spot where a rating goes, it has a probability for every digit 0 to 5. A normal run picks one. TheGavel reads all six and keeps the expected value, the most likely digit, and the spread. You get a continuous score, plus a confidence number, without the model writing anything.

The engine writes the JSON itself. The model only fills a few short text fields, and each rating is one forward pass with no sampling. That removes a whole class of failures: in my first experiment the model broke its own JSON on 4 of 44 postings, and that cannot happen when the engine owns the brackets.

## The setup

- **Hardware:** an NVIDIA DGX Spark (GB10, 128 GB unified memory).
- **Model:** Qwen3.6-35B-A3B in NVFP4 on SGLang, with prefix caching on.
- **Test set:** 44 real postings, each already scored by a full evaluation from a frontier model (Claude) that writes a detailed report, 12 of them strong fits (4.0 or higher).
- **What I measure:** Spearman correlation with those reference scores, recall of the strong ones at 4.0, false positives among the 32 weaker ones, and run-to-run noise.
- **How I timed it:** final timings were taken with everything else on the box stopped, back to back, one run each.

## Results

| Setup | Seconds per posting | Tokens written per posting | Spearman | Strong roles kept | False positives (of 32) |
|---|---|---|---|---|---|
| Baseline: the model writes a full report | 1.76 | 579 | 0.82 | 12/12 | 10 |
| TheGavel, same model, same server | **1.07** | **28** | 0.79 to 0.80 | 12/12 | 6 to 7 |

Same model, same server, same prompts. About a twentieth of the generated tokens, 1.65 times the throughput, every strong role kept, and fewer false positives on this run. Spearman moves 0.05 to 0.1 between runs for both setups, so 0.79 against 0.82 is a tie here; the fresh sets below are the stronger evidence.

**Why 1.65x and not 20x.** The work is prefill-bound. Across the 44 postings the baseline read 1.06 million prompt tokens and wrote 25,000. Cutting the writing helps, but most of the time is spent reading. If you judge with LLMs, measure prompt tokens per second, not decode tokens per second. The benchmarks that dominate the conversation are about decode.

## Then I tested it on postings it had never seen

Every number above comes from 44 postings I had tuned on, so a good result there could be luck. Back in September I had set aside every posting evaluated after a fixed date and never touched them. I drew fresh sets from that pool, and before scoring anything I wrote down what would count as a pass. Each set was scored twice by each setup.

**First fresh set, 140 postings.** The false-positive drop held up: TheGavel flagged about a third fewer weak roles as strong (11 against 16 of 40), and it ranked the postings far better (Spearman 0.81 against 0.65). But it missed my own bar on strong roles, keeping 32 of 40 at the 4.0 line against 34.5 for the baseline. The roles it lost had all scored just under 4.0. TheGavel reads a little low at the top, while the baseline, writing its own report, reads a little high.

**The fix, chosen only from postings already used: move TheGavel's line to 3.7,** which is the same as adding 0.3 to every score it gives. On a second fresh set of 180 postings it passed every bar:

| Second fresh set (means of two runs) | Strong kept (of 60) | False positives (of 60) | Bad misses | Spearman |
|---|---|---|---|---|
| Baseline at 4.0 | 56 | 24.5 | 4 | 0.74 |
| TheGavel at 3.7 | **57** | **17.5** | **2** | **0.77** |

About 30% fewer false positives, half the bad misses (a strong role scored 3.0 or lower), and no strong roles lost.

## The line that mattered was not the one I tested

The 4.0 line decides what counts as strong. But the screen I actually run is a pre-filter: it throws out postings below 2.0 so the full evaluation never spends time on them. At that line both judges had the same flaw. They threw out about 1 strong role in 22, and nearly every one was a location cap that misread my own policy, like a posting that allows remote work for people outside its office city, or a city on my approved list.

Three changes fixed it, each picked from postings already used and then checked on a fresh set:

- **A step-by-step location procedure in the prompt.** Check for a remote option first, then the approved cities, and use a hard no only when a posting clearly requires presence in an excluded city.
- **Ignore location caps the model writes itself.** Code already computes the location cap from the location rating, so the model's own cap is redundant and was the source of most errors.
- **Only cap when the model is confident.** This was the most useful find. When the cap wrongly hit a strong role, TheGavel's expected location rating sat between 1.5 and 2.1, a split between "hard no" and "maybe". On correct rejections it was almost always below 1.5. So the code cap now fires only at an expected rating of 1.5 or lower, and an unsure posting goes on to the full evaluation instead.

At the 2.0 line, against the screen I had been running:

| Fresh set | Strong roles kept, old screen | Strong roles kept, new screen | Share removed, old | Share removed, new |
|---|---|---|---|---|
| Third (180 postings) | 58 of 60 | **60 of 60** | 31.7% | 27.5% |
| Fourth (180), first shipping version | 57.5 of 60 | **58 of 60** | 25.0% | 21.7% |
| Fifth (156), with the confidence gate | 35 of 36 | 35 of 36 | 16.7% | **18.3%** |

The new screen never kept fewer strong roles than the old one, and with the confidence gate it removes as many weak postings as before. It now runs in my job-search tool as the default.

*The third set's new-screen row used my test harness's prompt; the fourth and fifth used the code that shipped. The fifth set is smaller because it used the last unused strong roles in the reserved pool.*

## Three things that surprised me

**1. The model has to be told the format, even when it never writes it.** My first live version read the ratings perfectly well, but the results were bimodal: probability piled up on 1 and on 5, and scores moved about a full point between identical runs. The model had never been told what layout it was "writing" or what the scale meant. Adding a short, engine-written format section to the prompt cut run-to-run movement to 0.05 points on my demo set.

**2. Free-text fields leak.** Some postings have a hard cap, like a location I can't work, and the cap overrides the average. I first made the cap a free-text field written before the ratings. In that version, the cap came out as "2.0" on strong roles, flipped between runs, and once the model copied my format placeholder into it. Recall fell to 9 and then 7 of 12. The fix was a "choice" field, read from probabilities like a rating (none, or 0 to 5), and placed after the ratings so the model has already judged the role. That brought recall back to 12/12.

**3. Deterministic mode was harder than one flag.** Even at temperature 0, scores moved about 0.3 points per posting between runs, because the server batches requests together and the floating-point sums change with the batch. SGLang has a deterministic flag. On my stack (a hybrid-attention MoE in NVFP4, speculative decoding, FlashInfer) it started, then crashed on the first real batch, and turning it on also disables the prefix cache.

## So I wrote a backend that is deterministic by construction

TheGavel has a second backend that runs a Hugging Face model in-process. The trick that makes it deterministic is boring on purpose. Every prompt is computed in fixed 256-token chunks counted from the start of the sequence, and the KV cache is only saved at chunk boundaries. A cached run and a cold run therefore do exactly the same arithmetic in the same order. It doesn't matter what else ran first, how many postings are in flight, or how busy the GPU is.

Result: two full runs over the 44 postings produce **byte-identical output files**. They stayed byte-identical across concurrency settings, cache states, and runs made while the GPU was busy with other work. The cache reused 98.8% of prompt tokens.

The catch is that this backend needs plain full attention. Qwen3.6 has linear-attention layers whose state can't be rewound, so I used a dense 4B model (Qwen3-4B-Instruct). The 4B judges clearly worse than the 35B: Spearman 0.67, with 10 of 12 strong roles kept. The engine guarantees the same answer every time, not a better answer. Model size still matters.

It also runs on small hardware. On an 8 GB laptop GPU (an RTX 4060 Laptop), Qwen3-1.7B scored all 44 postings at about 14 seconds each, and two full runs were byte-identical. A model that small judges a hard rubric poorly (Spearman 0.33), so the advice is simple: use the largest full-attention model that fits.

## Making it fast

The first native version took 9 seconds per posting on a quiet box, and the cause was not what I guessed. Profiling a single step on a 20,000-token cache showed:

- **Copying the KV cache was cheap** (66 ms), so that wasn't it.
- **Head expansion:** the model shares K/V heads across query heads, and the default attention path expanded them with a full copy in every layer, on every step. That was 36% of the GPU time.
- **Masked attention** fell back to a slower kernel. Switching to a lower-right causal mask with native grouped-query support gave the same numbers and was 10x faster per layer for short steps.
- **Cache growth:** while writing, the cache grew by concatenation, copying the whole cache for every new token. A preallocated buffer cut GPU time per token by 2.7x.
- **CPU overhead:** after that, the CPU was the bottleneck. A GPU sync in every layer and a 32,768-wide mask that my attention function ignored anyway made up most of it.

Quiet-box result: **9.0 down to 4.1 seconds per posting**, and the output files never changed by a single byte.

## Does it work on other engines? TensorFold as the test

TheGavel's first server mode needs two things from the engine: rating probabilities (logprobs), and a way to hand the model a half-written answer and have it carry on (`continue_final_message`). Plenty of engines don't document the second. So I added a mode that needs neither: TheGavel builds the prompt text itself and sends it to the plain `/v1/completions` endpoint that nearly every engine has. On SGLang it matched the chat mode, with results indistinguishable from one run to the next (Spearman 0.81 and 0.77, 12/12, 7 false positives).

Then I tried it on TensorFold 1.2.10, a newer engine with a DGX Spark recipe, using my own checkpoints.

- **Qwen3.6 NVFP4 loaded, but TensorFold reports no logprobs for it.** Both routes refuse the request, so TheGavel can't run there at all.
- **Nemotron 3.5 Lightning (30B-A3B, MLX 4-bit) does report logprobs,** and TheGavel's completions mode ran on it without changes. Two full runs came out byte-identical.
- **TensorFold ignores `continue_final_message`.** I sent the same half-written answer with the flag on and off and got identical requests and replies. The model started a fresh answer each time.

On the 41 postings both Nemotron setups scored:

| Setup | Spearman | Strong roles kept | False positives (of 29) | Malformed outputs | Tokens written per posting | Seconds per posting |
|---|---|---|---|---|---|---|
| Nemotron writes its own report | 0.67 | 8/12 | 5 | 3 of 44 | ~339 | 2.1 |
| Nemotron through TheGavel | 0.68 | 9/12 | 9 | 0 | ~24 | 33 |
| Qwen3.6 writes its own report (SGLang) | 0.83 | 12/12 | 9 | 0 | ~579 | 1.8 |
| Qwen3.6 through TheGavel (SGLang) | 0.79 | 12/12 | 7 | 0 | ~28 | 1.1 |

Three takeaways:

- **The model sets the ceiling, not the method.** With Nemotron, TheGavel and the normal way judge about equally well. Both sit well below Qwen.
- **TheGavel's sure wins carry over:** no broken output, and about a fourteenth of the tokens.
- **Speed depends on the engine's prompt cache, route by route.** TensorFold reused 93% of prompt tokens on its chat route and 0% on completions. TheGavel can only use completions there, so every step re-read about 21,000 tokens, and it ran 16 times slower than the normal way. The fix is on the engine side: honor `continue_final_message`, or cache prompts on completions.

*TensorFold timings ran at concurrency 4 and the SGLang ones at 8, single run each.*

## What I'd tell someone building an LLM judge

1. Read the rating probabilities instead of sampling them. You get a confidence number for free, and the model writes far less.
2. Let code write the structure. The model should fill values, not brackets.
3. Tell the model the format anyway.
4. Anything numeric should be read from probabilities like a rating, not written as free text. Put fields that depend on the ratings after them.
5. Your bottleneck is prefill. Benchmark prompt tokens per second.
6. If you need exactly the same score every time, don't count on the server's batching. Make the arithmetic independent of everything else that's running.
7. Before you pick an engine, check two things: does it give logprobs for your model, and does it cache prompts on the route you'll call? A feature that's "supported" can still depend on the model and on the route.
8. Use the confidence, not just the score. A hard cap should fire only when the model is sure. An unsure "maybe" is exactly the case to hand to a stronger judge.
9. Test on data you never tuned on, write the pass bar down first, and test at the threshold you actually use. My first fresh set looked fine at the line I had been watching and showed a real flaw at the line that decides what gets thrown out.

## Limits

The reference scores come from one judge (a full Claude evaluation), not from human labels. The fresh sets held 36 to 60 strong roles each, so differences of one or two roles are within noise, and each timing is a single run. The deterministic backend is only as good as the small model it can run today. The TensorFold results describe version 1.2.10 with two checkpoints, so they may change as the engine changes.

## Try it

Install and the two-minute demo are in the [README](../README.md).
