# Origin story

This is the brainstorming session that led to hAIvemind, lightly edited for readability.

---

## The spark

> "I call it hAIvemind. Basically you wire together as many free AI models as you can get your hands on. Send a prompt, the prompt gets bounced around between all the models until they all agree on the best output."

The idea came from an existing multi-model router where Claude Code teammates were wired to external AI providers (mostly free) to build, review, and critique code together.

## The framework

All models get the initial prompt. All models output their initial answer. All answers are concatenated. The concatenated result is given back to all models. All models critique, arbitrate, and rank what stays, goes, or changes.

```
Round 1: DIVERGE
Prompt --> Model A --> Answer A
       --> Model B --> Answer B
       --> Model C --> Answer C

Round 2: SHARE
[Answer A + B + C] --> All Models

Round 3: CONVERGE
Each model critiques/ranks/filters

Round 4+: SYNTHESIZE
Loop until stable
```

## No arbiter

> "I don't want an arbiter. It won't work with free models."

The whole reason the multi-model router works so well is you get different outputs from each model. You can give 2 different models the same code and same prompt — one finds a bug but one doesn't. Do that x5 and you cover so many bases.

No single free model is as good as a paid frontier model. But 5 free models critiquing each other might be. The intelligence emerges from the process, not any individual model.

## Shared memory, not message passing

> "Think differently — Claude Teams — the models all talk to each other through shared files — that's the way forward."

```
+-----------------------------+
|      SHARED MEMORY SPACE    |
|                             |
|  - Original prompt          |
|  - All responses            |
|  - All critiques            |
|  - Running "document"       |
|    that models edit/evolve  |
+---------+-------------------+
          |
    +-----+-----+-----+-----+
    v     v     v     v     v
  Gemini Llama Mistral Cohere Qwen
    |     |     |     |     |
    +-----+-----+-----+-----+
          |
          v
   All write back to shared space
   Read what others wrote
   React, critique, build on it
          |
          v
   Loop until stable
```

The key insight: it's not about finding the best model. It's about coverage. Model A catches bug X but misses Y. Model B catches Y but misses X. Together they catch both. No arbiter needed — the shared space accumulates knowledge.

## The locked room

> "I think of the real world analogue as a room with 5 people in it. You tell them they need to solve a problem to leave the room."

Social pressure as a consensus mechanism. You're not programming consensus logic — you're giving models human social dynamics through prompting.

**The room rules:**
1. You are in a room with other agents
2. You all received the same problem
3. Nobody leaves until you ALL agree
4. The answer must be CORRECT — agreeing on a wrong answer doesn't open the door
5. Read what everyone else has said
6. Challenge things you think are wrong
7. Defend your position — but know when you're outnumbered
8. The door opens when the conversation stops

## Fluid roles

> "I think the prompt should make them flexible. 'Approach from all angles, be the adversarial critique or the stubborn expert when required.' Maybe have a list of roles and tell them to move freely between them."

Don't cast actors. Let them improvise.

- **THE CRITIC**: Attack weak reasoning
- **THE BUILDER**: Improve on others' ideas
- **THE EXPERT**: Go deep on what you know
- **THE SKEPTIC**: "Are we sure about this?"
- **THE SYNTHESIZER**: Pull threads together
- **THE BREAKER**: Try to disprove consensus
- **THE CONCEDER**: Know when to let go

Fixed roles create predictable patterns that models will game. Fluid roles mean a model might start as builder then switch to critic when it spots a flaw mid-thought.

## The spectator appeal

> "Honestly part of the appeal is watching AI talk to each other. I've watched Gemini and GPT go back and forth on a coding task and it's mesmerising."

This is two things at once:
- **The tool**: Give me the best answer free models can produce
- **The theatre**: Let me watch AIs locked in a room arguing about my question

## The minimum viable product

> "tmux could take care of that"

The entire framework is:
- One system prompt (the room rules)
- A shared space (the run directory)
- A loop (keep going until convergence)
- Free models (as many as you can wire up)

That's it. The prompt engineering IS the product.

---

*This conversation happened the night before the first working prototype was built.*
