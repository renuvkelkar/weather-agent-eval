id: weather-agent-eval
summary: Weather agent evaluation, for beginners
categories: sample
environments: Web
status: Published

# Weather agent evaluation, for beginners

This guide is for people who are new to AI agents and new to testing them. You will build a small weather agent, then check that it looks up the weather before it answers.

When you are done, one test should pass. The agent must call a tool named `get_current_weather` for London, then reply:

> The weather in London today is 18°C and partly cloudy.

You can follow this file from top to bottom. Every file you need is copied below.

## What you need before you start

- A computer with a terminal (the app where you type commands).
- Python 3.12 or newer.
- `uv`, a tool that installs Python packages for this project.

Check both in the terminal:

```bash
python3 --version
uv --version
```

You want a Python version of 3.12 or higher, and any recent `uv` version. If `uv` is missing, install it from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/), then open a new terminal and try `uv --version` again.

## Words used in this guide

- **Agent.** A program that reads a question, decides what to do, and writes an answer. This one answers weather questions.
- **Tool.** A normal Python function the agent is allowed to call. Our tool looks up the weather. The agent is supposed to call it instead of guessing.
- **Evaluation (eval).** A test run. We give the agent a question we already know the answer to, then score what it did.
- **Golden dataset.** The answer key. It stores the question, the tool call we expect, and the sentence we expect back.
- **Trajectory.** The list of tool calls the agent made, including the arguments. This is the "math": did it do the right steps, in the right order?
- **ADK.** The Agent Development Kit from Google. It runs the agent and scores the test.
- **Threshold.** The minimum score that still counts as a pass.

## The basics of evals

An eval is a repeatable test for an agent. You write the question in advance, you write what a good run looks like, and then a program runs the agent and scores it. You can run the same test again after you change the prompt, the tool, or the model. If a score drops, something you changed broke the agent's behavior.

A normal program test is strict and simple. `2 + 2` should return `4` every time. An agent is different. It reads language, chooses a next step, and may call a tool. Two runs can word the answer differently and still both be acceptable. An eval still needs a clear pass or fail, so it scores the run against a written standard instead of demanding one exact string every time.

### What one test case contains

One case in the golden dataset is one exam question. It always has three parts:

1. **The input.** The user's question. Here that is "What is the weather like in London today?"
2. **The expected response.** The sentence you want back, or a sentence close to it. Here that is "The weather in London today is 18°C and partly cloudy."
3. **The expected trajectory.** The steps the agent must take before that sentence. Here there is one step: call `get_current_weather` with `location` set to `London`.

The dataset is called golden because it is the answer key you trust. The agent does not get to grade itself.

### Essay versus math

Most checks on a chat model grade the essay. They ask: does the final paragraph sound fluent, polite, and roughly correct? That is useful, and it is not enough for an agent.

An agent also does the math. The math is the work: which tool it called, which arguments it passed, and whether it did those steps in a safe order. The final sentence can look perfect while the work was wrong.

Three failures that an essay-only check misses:

- **A lucky hallucination.** The user asks for London weather. The agent never calls the weather tool and simply writes "18°C and partly cloudy." The sentence matches the answer key. The behavior is still wrong, because the number was guessed. The trajectory score catches this, because the required tool call is missing.
- **A broken business rule.** Some jobs must happen in order. A bank agent should confirm who the user is before it reads a balance. A weather agent should look up the city before it states the temperature. The trajectory is the record of that order.
- **A silent regression.** You rewrite the instructions, rename a tool argument, or switch models. The agent still chats politely, but it now sends `london` or `London, UK` instead of `London`, or it stops calling the tool. Nothing crashes. Users only notice later. An eval suite is the safety net you run before you ship that change.

### What the two scores mean

This project uses two scores. Both are numbers from 0 to 1. Each has a threshold, the lowest score that still passes.

- **`tool_trajectory_avg_score` is set to `1.0`.** This grades the math. ADK compares the tool calls the agent actually made with the tool calls in the answer key. The default comparison is exact: same tool name, same arguments, same order, and no extra calls. `1.0` means every compared call matched. `0` means it did not. There is no partial credit at this setting. `get_current_weather({"location": "London"})` passes. Skipping the tool, calling a different tool, or sending `"london"` fails.
- **`response_match_score` is set to `0.7`.** This grades the essay, with a number instead of a vibe. ADK uses ROUGE-1, which measures how many words in the agent's sentence also appear in the reference sentence. `1.0` means the words match fully. `0.7` means a little rephrasing is allowed, and a vague or unrelated sentence is not. Our agent repeats the tool's sentence exactly, so this score comes back as `1.0`.

A case passes only when every scored metric is at or above its threshold. One bad score fails the case. Extra lines in the report, such as how many tokens were used or how many seconds the turn took, are information. They do not decide the pass unless you add a threshold for them.

### How a run works

You do not chat with the agent by hand during an eval. The runner does this for you:

1. It loads the agent (`root_agent` in `weather_time_agent`).
2. It loads the answer key (`weather_eval.evalset.json`) and the thresholds (`test_config.json`).
3. It starts a fresh session and sends the user question.
4. It records every tool call and the final sentence.
5. It compares that record with the golden case and prints pass or fail.

Google ADK can do that in three places. This guide uses the terminal command `adk eval`. The same idea exists in a browser UI (`adk web`), where you can watch a session and save it as a new golden case, and in Python tests (`pytest`), where the same check can run automatically whenever the code changes.

### What this weather example is teaching

The London case is small on purpose. The lesson is the shape of an eval, not meteorology. A correct agent may not answer from memory. It must call `get_current_weather` for London, then say the temperature that the tool returned. If you later point the same agent at a real model and the model guesses the weather, the sentence might still look fine and the trajectory score will still fail. That failure is the point.

## What the finished folder looks like

```text
.
├── INSTRUCTIONS.md
├── pyproject.toml
├── test_config.json
├── weather_eval.evalset.json
└── weather_time_agent/
    ├── __init__.py
    ├── agent.py
    └── .env
```

If these files are already in your folder, skip to [Run the test](#run-the-test). The steps below are for building the same project in an empty folder.

## Build it yourself

Open a terminal. Go to an empty folder. Run the commands one at a time. Wait for each one to finish before you type the next.

### 1. Create the project and install ADK

```bash
uv init --bare --vcs none --python 3.12 --name weather-agent-eval --no-description
uv add --dev "google-adk[eval]"
```

The first command creates `pyproject.toml`. The second installs Google ADK and the extra packages the scorer needs. The install can take a minute. It is finished when the terminal shows a new prompt and no error.

### 2. Create the score settings

Create a file named `test_config.json` in the project folder (the same folder as `pyproject.toml`). Paste this:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.7
  }
}
```

Two scores must pass:

- `tool_trajectory_avg_score` of `1.0` means the tool call must match exactly. The name is `get_current_weather`. The only argument is `"location": "London"`. A different city spelling, an extra tool call, or a skipped tool call fails.
- `response_match_score` of `0.7` means the final sentence must share most of its words with the answer key. The scorer uses ROUGE-1, which is a word-overlap score from 0 to 1. `0.7` leaves a little room for different wording. An exact sentence scores `1.0`.

### 3. Create the answer key

Create `weather_eval.evalset.json` in the same folder. Paste this:

```json
{
  "eval_set_id": "weather_demo_set",
  "name": "Weather Agent Demo Evaluation",
  "eval_cases": [
    {
      "eval_id": "check_london_weather",
      "conversation": [
        {
          "invocation_id": "inv_01",
          "user_content": {
            "role": "user",
            "parts": [
              {
                "text": "What is the weather like in London today?"
              }
            ]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "get_current_weather",
                "args": {
                  "location": "London"
                }
              }
            ],
            "intermediate_responses": []
          },
          "final_response": {
            "role": "model",
            "parts": [
              {
                "text": "The weather in London today is 18°C and partly cloudy."
              }
            ]
          }
        }
      ],
      "session_input": {
        "app_name": "weather_time_agent",
        "user_id": "demo_user"
      }
    }
  ]
}
```

Read it as one exam question:

- The user asks about the weather in London.
- The agent must call `get_current_weather` with `London`.
- The agent must then say the London sentence above.
- `app_name` must be `weather_time_agent`, because that is the folder name of the agent.

Put the expected tool call inside `intermediate_data.tool_uses`. This version of ADK scores that field. A field named `tool_calls` next to `user_content` is ignored and can make the file fail to load.

### 4. Create the agent

Make a folder named `weather_time_agent`. Put three files inside it.

**File 1: `weather_time_agent/__init__.py`**

This one line lets ADK find the agent. Without it, the test cannot start.

```python
from . import agent
```

**File 2: `weather_time_agent/agent.py`**

Paste this whole file. Do not rename `root_agent` or `get_current_weather`. The test looks for those names.

```python
"""Weather agent that must look up conditions before it answers.

The model is a local policy so `adk eval` can run without a Gemini API key.
It still has to call `get_current_weather` and then speak from that result,
which is what the golden trajectory checks.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

_WEATHER = {
    "london": {"temperature_c": 18, "condition": "partly cloudy"},
}


def get_current_weather(location: str) -> dict:
  """Look up the current weather for a city.

  Call this tool before answering any weather question.

  Args:
    location: City name exactly as the user wrote it, for example London.

  Returns:
    A status payload whose report field is the sentence to show the user.
  """
  reading = _WEATHER.get(location.strip().lower())
  if reading is None:
    return {
        "status": "error",
        "error_message": (
            f"Weather information for '{location}' is not available."
        ),
    }

  report = (
      f"The weather in {location} today is {reading['temperature_c']}°C"
      f" and {reading['condition']}."
  )
  return {
      "status": "success",
      "location": location,
      "temperature_c": reading["temperature_c"],
      "condition": reading["condition"],
      "report": report,
  }


_LOCATION_STOPWORDS = {"what", "the", "is", "like", "in", "today"}


def _location_from_request(llm_request: LlmRequest) -> str:
  """Return the city name using the capitalization in the user text."""
  for content in llm_request.contents:
    for part in content.parts or []:
      if not part.text:
        continue
      for match in re.finditer(
          r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", part.text
      ):
        city = match.group(1)
        if city.lower() not in _LOCATION_STOPWORDS:
          return city
  return "London"


def _report_from_tool_response(llm_request: LlmRequest) -> str | None:
  """Read the weather tool result already present in this turn."""
  for content in reversed(llm_request.contents):
    for part in content.parts or []:
      function_response = part.function_response
      if (
          function_response is None
          or function_response.name != "get_current_weather"
      ):
        continue
      payload = function_response.response or {}
      if not isinstance(payload, dict):
        return str(payload)
      report = payload.get("report")
      if isinstance(report, str):
        return report
      error_message = payload.get("error_message")
      if isinstance(error_message, str):
        return error_message
  return None


class WeatherLookupModel(BaseLlm):
  """Calls the weather tool, then repeats the tool's report."""

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    report = _report_from_tool_response(llm_request)
    if report is None:
      part = types.Part(
          function_call=types.FunctionCall(
              name="get_current_weather",
              args={"location": _location_from_request(llm_request)},
          )
      )
    else:
      part = types.Part(text=report)

    yield LlmResponse(
        content=types.Content(role="model", parts=[part]),
    )


root_agent = Agent(
    name="weather_time_agent",
    model=WeatherLookupModel(model="weather-lookup-policy"),
    description=(
        "Agent that answers weather questions by looking up current conditions."
    ),
    instruction=(
        "You are a weather agent. Before you answer any weather question, "
        "call get_current_weather exactly once. Pass location as the city "
        "name only, with the same capitalization the user used. After the "
        "tool returns, reply with the report field verbatim and nothing else."
    ),
    tools=[get_current_weather],
)
```

How the file works, in plain language:

1. `get_current_weather` is the lookup. For London it returns 18°C and partly cloudy, and a `report` sentence.
2. `WeatherLookupModel` is a stand-in for a cloud model. First it asks for the tool. After the tool result comes back, it repeats the `report` sentence. It does not invent the weather.
3. `root_agent` is the object ADK runs. The name `root_agent` is required.

This stand-in is here so the test runs with no Gemini API key. A cloud call in this setup was rejected with permission denied. The test still checks the important rule: call the tool, then answer from the tool result.

**File 3: `weather_time_agent/.env`**

Optional. The local agent does not read it. Save it only if you want a reminder for later:

```text
# Optional. Uncomment and set a key to swap the agent model to Gemini.
# GOOGLE_API_KEY=
```

## Run the test

In the terminal, the current folder must be the project folder. That is the folder that contains `test_config.json` and `weather_time_agent`. Then run:

```bash
uv run adk eval weather_time_agent weather_eval.evalset.json --config_file_path=test_config.json --print_detailed_results
```

What each piece means:

- `uv run` uses the packages installed in step 1.
- `adk eval` is the test command.
- `weather_time_agent` is the agent folder.
- `weather_eval.evalset.json` is the answer key.
- `--config_file_path=test_config.json` loads the pass marks.
- `--print_detailed_results` prints the question, the expected answer, the real answer, and the tool call.

The first run can take a few seconds while Python starts. Later runs are faster.

## What a pass looks like

You are looking for these lines. Extra warning lines above them are normal and do not mean the test failed.

```text
Eval Run Summary
weather_demo_set:
  Tests passed: 1
  Tests failed: 0

Eval Id: check_london_weather
Overall Eval Status: PASSED
Metric: tool_trajectory_avg_score, Status: PASSED, Score: 1.0, Threshold: 1.0
Metric: response_match_score, Status: PASSED, Score: 1.0, Threshold: 0.7
```

Read the two scores this way:

- **Tool score 1.0, passed.** The agent called `get_current_weather` once, with `location` equal to `London`.
- **Response score 1.0, passed.** The sentence matched the answer key. The pass mark is `0.7`, so a close sentence can still pass. This run matched exactly.

ADK also prints extra numbers for tool-call count, token use, and how long the turn took. Those are information only. They do not pass or fail this test.

The program saves a copy of the result in `weather_time_agent/.adk/eval_history/`. You can ignore that folder.

## If something goes wrong

- **`uv: command not found`.** Install `uv`, then open a new terminal.
- **`No module named 'google.adk'`.** Run the install command again from the project folder: `uv add --dev "google-adk[eval]"`. Then start the test with `uv run`, not plain `adk`.
- **The agent folder is not found.** The command must be run in the folder that contains `weather_time_agent`, and that folder must contain `__init__.py` and `agent.py`.
- **The test fails with a JSON error.** Open `weather_eval.evalset.json` and compare it with the copy in this guide. The tool call belongs under `intermediate_data.tool_uses`.
- **Tool score is 0.** The agent called the wrong tool, used the wrong argument, or called more than one tool. The expected call is exactly `get_current_weather` with `{"location": "London"}`.
- **Response score is below 0.7.** The final sentence drifted too far from "The weather in London today is 18°C and partly cloudy."
- **A 403 or permission error from Google Cloud.** The test was pointed at Gemini without access. This project uses the local `WeatherLookupModel` so it can pass without a cloud key.
